package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// DriveUploadRequest defines the payload schema sent to the Google Apps Script Web App
type DriveUploadRequest struct {
	FolderID string `json:"folderId"`
	FileName string `json:"fileName"`
	Content  string `json:"content"`
	MimeType string `json:"mimeType"`
}

// uploadToDrive uploads a local file (text or binary) to Google Drive via the Google Apps Script Web App Gateway.
// Authentication is handled transparently by the Web App executing as the user.
func uploadToDrive(ctx context.Context, localFilePath, driveFileName, mimeType string) error {
	webAppURL := os.Getenv("GOOGLE_DRIVE_WEB_APP_URL")
	if webAppURL == "" {
		return fmt.Errorf("the GOOGLE_DRIVE_WEB_APP_URL environment variable is not defined in .env")
	}

	// Read the local file content
	contentBytes, err := os.ReadFile(localFilePath)
	if err != nil {
		return fmt.Errorf("failed to read local file %s: %w", localFilePath, err)
	}

	var contentStr string
	if mimeType == "audio/wav" {
		// Encode binary files as Base64 so they can be securely transmitted inside JSON payload
		contentStr = base64.StdEncoding.EncodeToString(contentBytes)
	} else {
		contentStr = string(contentBytes)
	}

	folderID := os.Getenv("GOOGLE_DRIVE_FOLDER_ID")
	reqBody := DriveUploadRequest{
		FolderID: folderID,
		FileName: driveFileName,
		Content:  contentStr,
		MimeType: mimeType,
	}

	jsonBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal upload request: %w", err)
	}

	fmt.Printf("Uploading file '%s' (%s) to Google Drive via Apps Script...\n", driveFileName, mimeType)

	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "POST", webAppURL, bytes.NewBuffer(jsonBytes))
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Standard Go client automatically follows 302 redirects returned by Apps Script Web Apps
	// 60-second timeout allows base64 decoding and file creation of audio files on Google servers
	client := &http.Client{
		Timeout: 60 * time.Second,
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to upload via Apps Script (status %d): %s", resp.StatusCode, string(respBody))
	}

	// Check response content if it indicates success
	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err == nil {
		if status, ok := result["status"].(string); ok {
			if status == "success" {
				fmt.Printf("Upload to Google Drive completed successfully! File ID: %v\n", result["fileId"])
				return nil
			} else if msg, ok := result["message"].(string); ok {
				return fmt.Errorf("Apps Script execution error: %s", msg)
			}
		}
	}

	// Fallback check if response contains success text directly
	if bytes.Contains(respBody, []byte("success")) {
		fmt.Println("Upload to Google Drive completed successfully!")
		return nil
	}

	return fmt.Errorf("unexpected response from Apps Script: %s", string(respBody))
}
