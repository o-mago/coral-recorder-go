package main

import (
	"context"
	"fmt"
	"os"

	"google.golang.org/api/drive/v3"
	"google.golang.org/api/option"
)

// uploadToDrive uploads a local file to Google Drive.
// If the GOOGLE_DRIVE_FOLDER_ID environment variable is set, the file will be uploaded into that folder.
// Authentication is done using the Service Account JSON file pointed to by the GOOGLE_APPLICATION_CREDENTIALS variable.
func uploadToDrive(ctx context.Context, localFilePath, driveFileName string) error {
	credsPath := os.Getenv("GOOGLE_APPLICATION_CREDENTIALS")
	if credsPath == "" {
		credsPath = os.Getenv("GOOGLE_APPLICATION_CREDENTIAL")
	}
	if credsPath == "" {
		return fmt.Errorf("the GOOGLE_APPLICATION_CREDENTIALS environment variable is not defined. " +
			"To upload to Google Drive, set this variable to the path of your JSON credentials file")
	}

	// Initialize the Google Drive service using the provided credentials
	srv, err := drive.NewService(ctx, option.WithCredentialsFile(credsPath))
	if err != nil {
		return fmt.Errorf("failed to initialize Google Drive service: %w", err)
	}

	// Open the local file
	file, err := os.Open(localFilePath)
	if err != nil {
		return fmt.Errorf("failed to open local file %s for upload: %w", localFilePath, err)
	}
	defer file.Close()

	// Define the file metadata
	driveFile := &drive.File{
		Name:     driveFileName,
		MimeType: "text/markdown",
	}

	// If a destination folder ID was provided, associate the file with it
	folderID := os.Getenv("GOOGLE_DRIVE_FOLDER_ID")
	if folderID != "" {
		driveFile.Parents = []string{folderID}
		fmt.Printf("Uploading file to the Drive folder with ID: %s\n", folderID)
	} else {
		fmt.Println("No GOOGLE_DRIVE_FOLDER_ID defined. The file will be created in the Drive root.")
	}

	fmt.Printf("Uploading file '%s' to Google Drive...\n", driveFileName)
	_, err = srv.Files.Create(driveFile).Media(file).Do()
	if err != nil {
		return fmt.Errorf("failed to create file on Google Drive: %w", err)
	}

	fmt.Println("Upload to Google Drive completed successfully!")
	return nil
}
