package main

import (
	"embed"
	"fmt"
	"io/fs"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

//go:embed main.py requirements.txt assets/*
var embeddedFiles embed.FS

var (
	version   = "0.0.0"
	branch    = "unknown"
	commit    = "unknown"
	buildDate = "unknown"
)

const (
	InternalPort = "8202"
	PublicPort   = "8200"
	ServiceName  = "dex-tts-service"
)

func main() {
	// Silence unused variable warnings by referencing them
	_ = branch
	_ = commit
	_ = buildDate

	if len(os.Args) > 1 && os.Args[1] == "version" {
		fmt.Println(version)
		return
	}

	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Printf("Starting %s Wrapper...", ServiceName)

	// 1. Setup Environment
	homeDir, err := os.UserHomeDir()
	if err != nil {
		log.Fatalf("Failed to get home directory: %v", err)
	}
	serviceDir := filepath.Join(homeDir, "Dexter", "services", ServiceName)
	venvDir := filepath.Join(serviceDir, "venv")

	// Ensure service directory exists
	if err := os.MkdirAll(serviceDir, 0755); err != nil {
		log.Fatalf("Failed to create service directory: %v", err)
	}

	// Extract embedded files
	log.Println("Extracting embedded files...")
	if err := extractFiles(serviceDir); err != nil {
		log.Fatalf("Failed to extract files: %v", err)
	}

	// 2. Python Environment Setup
	pythonExecutable := "python"
	pipExecutable := "pip"
	if runtime.GOOS == "windows" {
		pythonExecutable = "python.exe"
		pipExecutable = "pip.exe"
	}

	venvPython := filepath.Join(venvDir, "bin", pythonExecutable)
	venvPip := filepath.Join(venvDir, "bin", pipExecutable)
	if runtime.GOOS == "windows" {
		venvPython = filepath.Join(venvDir, "Scripts", pythonExecutable)
		venvPip = filepath.Join(venvDir, "Scripts", pipExecutable)
	}

	if _, err := os.Stat(venvPython); os.IsNotExist(err) {
		log.Println("Creating virtual environment...")
		// Try compatible Python versions first
		pythonCmd := "python3"
		compatibleVersions := []string{"python3.10", "python3.11", "python3"}
		for _, v := range compatibleVersions {
			if _, err := exec.LookPath(v); err == nil {
				pythonCmd = v
				// Check version if it's just "python3"
				if v == "python3" {
					out, _ := exec.Command(v, "--version").Output()
					if strings.Contains(string(out), "3.13") || strings.Contains(string(out), "3.12") {
						log.Printf("Skipping %s as it is too new for TTS library", v)
						continue
					}
				}
				log.Printf("Using %s to create virtual environment", v)
				break
			}
		}

		cmd := exec.Command(pythonCmd, "-m", "venv", venvDir)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Fatalf("Failed to create venv: %v. Ensure python3 is installed.", err)
		}
	}

	// Install Dependencies
	log.Println("Checking dependencies (this may take a while for large packages like torch)...")
	// We force install/upgrade from requirements.txt to ensure consistency
	cmd := exec.Command(venvPip, "install", "-r", filepath.Join(serviceDir, "requirements.txt"))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		log.Printf("Warning: pip install failed: %v", err)
		// We continue, as it might be a transient network issue or already installed
	}

	// 3. Start Python Service
	log.Printf("Starting Python backend on port %s...", InternalPort)
	pyCmd := exec.Command(venvPython, "main.py")
	pyCmd.Dir = serviceDir
	// Pass environment variables, ensuring PORT is set
	env := os.Environ()
	env = append(env, "PORT="+InternalPort)
	env = append(env, "HOST=127.0.0.1")
	// Important: Unbuffer output
	env = append(env, "PYTHONUNBUFFERED=1")
	// Inject Version Info
	env = append(env, "DEX_VERSION="+version)
	env = append(env, "DEX_BRANCH="+branch)
	env = append(env, "DEX_COMMIT="+commit)
	env = append(env, "DEX_BUILD_DATE="+buildDate)
	pyCmd.Env = env

	pyCmd.Stdout = os.Stdout
	pyCmd.Stderr = os.Stderr

	if err := pyCmd.Start(); err != nil {
		log.Fatalf("Failed to start python service: %v", err)
	}

	// Ensure we kill the subprocess on exit
	defer func() {
		if pyCmd.Process != nil {
			_ = pyCmd.Process.Kill()
		}
	}()

	// Wait for port to be open
	if err := waitForPort("127.0.0.1:"+InternalPort, 30*time.Second); err != nil {
		log.Fatalf("Python service failed to start listening: %v", err)
	}

	// 4. Start Proxy
	targetURL, _ := url.Parse("http://127.0.0.1:" + InternalPort)
	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		proxy.ServeHTTP(w, r)
	})

	log.Printf("Dexter TTS Service (Go Wrapper) listening on :%s", PublicPort)
	if err := http.ListenAndServe(":"+PublicPort, nil); err != nil {
		log.Fatalf("Proxy server failed: %v", err)
	}
}

func extractFiles(destDir string) error {
	return fs.WalkDir(embeddedFiles, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if path == "." {
			return nil
		}

		fullPath := filepath.Join(destDir, path)
		if d.IsDir() {
			return os.MkdirAll(fullPath, 0755)
		}

		data, err := embeddedFiles.ReadFile(path)
		if err != nil {
			return err
		}
		return os.WriteFile(fullPath, data, 0644)
	})
}

func waitForPort(addr string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 1*time.Second)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("timeout waiting for %s", addr)
}
