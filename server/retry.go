package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// PendingSession holds the metadata needed to retry transcription
type PendingSession struct {
	Timestamp string   `json:"timestamp"`
	Events    []string `json:"events"`
}

var triggerRetryChan = make(chan struct{}, 1)

// triggerUploadRetry sends a non-blocking signal to the retry worker
func triggerUploadRetry() {
	select {
	case triggerRetryChan <- struct{}{}:
	default:
		// Signal already queued
	}
}

// startRetryLoop starts the background worker goroutine
func startRetryLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	log.Println("Background upload retry loop started.")

	// Run once immediately on startup to catch any leftover recordings from a power cycle
	processQueue(ctx)

	for {
		select {
		case <-ctx.Done():
			log.Println("Background upload retry loop stopped.")
			return
		case <-ticker.C:
			processQueue(ctx)
		case <-triggerRetryChan:
			processQueue(ctx)
		}
	}
}

// processQueue scans the folder, sorts pending files, and attempts upload one by one
func processQueue(ctx context.Context) {
	queueDir := getQueueDir()
	files, err := os.ReadDir(queueDir)
	if err != nil {
		log.Printf("Queue worker: failed to read directory: %v", err)
		return
	}

	var wavFiles []string
	for _, f := range files {
		if !f.IsDir() && strings.HasPrefix(f.Name(), "pending_") && strings.HasSuffix(f.Name(), ".wav") {
			wavFiles = append(wavFiles, filepath.Join(queueDir, f.Name()))
		}
	}

	if len(wavFiles) == 0 {
		return
	}

	// Mark uploading in progress so LED displays blue
	SetUploadingProgress(true)
	defer SetUploadingProgress(false)

	// Set reprocessing in progress so LED blinks blue
	SetReprocessingProgress(true)
	defer SetReprocessingProgress(false)

	// Sort chronologically (alphabetical order of pending_YYYY-MM-DD_HH-MM-SS.wav is chronological)
	sort.Strings(wavFiles)

	log.Printf("Queue worker: found %d pending recording(s) to process.", len(wavFiles))

	for _, wavFile := range wavFiles {
		log.Printf("Queue worker: uploading %s...", wavFile)

		// Try to read the associated JSON events file
		jsonPath := strings.TrimSuffix(wavFile, ".wav") + ".json"
		var events []string
		if jsonBytes, err := os.ReadFile(jsonPath); err == nil {
			var metadata PendingSession
			if err := json.Unmarshal(jsonBytes, &metadata); err == nil {
				events = metadata.Events
			} else {
				log.Printf("Queue worker: warning: failed to parse metadata %s: %v", jsonPath, err)
			}
		}

		// Try uploading
		err = uploadAndTranscribe(ctx, wavFile, events)
		if err != nil {
			log.Printf("Queue worker: failed to process %s: %v. Will retry later.", wavFile, err)
			// Trigger LED to start blinking (or ensure it is blinking)
			TriggerLEDUpdate()
			// Abort processing subsequent sessions to avoid parallel timeouts or rate limits when internet is down
			return
		}

		log.Printf("Queue worker: successfully processed and deleted %s", wavFile)
		// Trigger LED update (if queue becomes empty, it transitions back to ready/green)
		TriggerLEDUpdate()
	}
}
