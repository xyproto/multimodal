// Package multimodal abstracts the genai multimodal prompt building
package multimodal

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"cloud.google.com/go/vertexai/genai"
)

// MultiModal represents multimodal prompt parts + configuration
type MultiModal struct {
	modelName   string
	temperature float32
	parts       []genai.Part
	trim        bool
	verbose     bool
}

// New creates a new MultiModal instance with a specified model name and temperature,
// initializing it with default values for parts, trim, and verbose settings.
func New(modelName string, temperature float32) *MultiModal {
	parts := make([]genai.Part, 0)
	return &MultiModal{modelName, 0.4, parts, true, false}
}

// SetVerbose updates the verbose logging flag of the MultiModal instance,
// allowing for more detailed output during operations.
func (mm *MultiModal) SetVerbose(verbose bool) {
	mm.verbose = verbose
}

// SetTrim updates the trim flag of the MultiModal instance,
// controlling whether the output is trimmed for whitespace.
func (mm *MultiModal) SetTrim(trim bool) {
	mm.trim = trim
}

// AddImage reads an image from a file, prepares it for processing,
// and adds it to the list of parts to be used by the model.
// It supports verbose logging of operations if enabled.
func (mm *MultiModal) AddImage(filename string) error {
	imageBytes, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	if mm.verbose {
		fmt.Printf("Read %d bytes from %s.\n", len(imageBytes), filename)
	}
	ext := strings.TrimPrefix(filepath.Ext(filename), ".")
	if ext == "jpg" {
		ext = "jpeg"
	}
	if mm.verbose {
		fmt.Printf("Using ext type: %s\n", ext)
	}
	img := genai.ImageData(ext, imageBytes)
	if mm.verbose {
		fmt.Printf("Prepared an image blob: %T\n", img)
	}
	mm.parts = append(mm.parts, img)
	return nil
}

// MustAddImage is a convenience function that adds an image to the MultiModal instance,
// terminating the program if adding the image fails.
func (mm *MultiModal) MustAddImage(filename string) {
	if err := mm.AddImage(filename); err != nil {
		log.Fatalln(err)
	}
}

// AddURI adds a file part to the MultiModal instance from a Google Cloud URI,
// allowing for integration with cloud resources directly.
// Example URI: "gs://generativeai-downloads/images/scones.jpg"
func (mm *MultiModal) AddURI(URI string) {
	mm.parts = append(mm.parts, genai.FileData{
		MIMEType: mime.TypeByExtension(filepath.Ext(URI)),
		FileURI:  URI,
	})
}

// AddURL downloads the file from the given URL, identifies the MIME type,
// and adds it as a genai.Part.
func (mm *MultiModal) AddURL(URL string) error {
	resp, err := http.Get(URL)
	if err != nil {
		return fmt.Errorf("failed to download the file from URL: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read the response body: %v", err)
	}
	mimeType := resp.Header.Get("Content-Type")
	if mimeType == "" {
		return fmt.Errorf("could see a Content-Type header for the given URL: %s", URL)
	}
	if mm.verbose {
		fmt.Printf("Downloaded %d bytes with MIME type %s from %s.\n", len(data), mimeType, URL)
	}
	fileData := genai.Blob{
		MIMEType: mimeType,
		Data:     data,
	}
	mm.parts = append(mm.parts, fileData)
	return nil
}

// AddData adds arbitrary data with a specified MIME type to the parts of the MultiModal instance.
func (mm *MultiModal) AddData(mimeType string, data []byte) {
	fileData := genai.Blob{
		MIMEType: mimeType,
		Data:     data,
	}
	mm.parts = append(mm.parts, fileData)
}

// AddText adds a textual part to the MultiModal instance.
func (mm *MultiModal) AddText(prompt string) {
	mm.parts = append(mm.parts, genai.Text(prompt))
}

// SubmitToClient sends all added parts to the specified Vertex AI model for processing,
// returning the model's response. It supports temperature configuration and response trimming.
func (mm *MultiModal) SubmitToClient(ctx context.Context, client *genai.Client) (string, error) {
	// Configure the model
	model := client.GenerativeModel(mm.modelName)
	model.SetTemperature(mm.temperature)
	// Then pass in the parts and generate a response
	res, err := model.GenerateContent(ctx, mm.parts...)
	if err != nil {
		return "", fmt.Errorf("unable to generate contents: %v", err)
	}
	// Then examine the reponse
	if len(res.Candidates) == 0 || len(res.Candidates[0].Content.Parts) == 0 {
		return "", errors.New("empty response from model")
	}
	// And return the result as a string
	result := fmt.Sprintf("%s\n", res.Candidates[0].Content.Parts[0])
	if mm.trim {
		return strings.TrimSpace(result), nil
	}
	return result, nil
}

// Submit sends all added parts to the specified Vertex AI model for processing,
// returning the model's response. It supports temperature configuration and response trimming.
// This function creates a temporary client, and is not meant to be used within Google Cloud (use SubmitToClient instead).
func (mm *MultiModal) Submit(projectID, location string) (string, error) {
	ctx := context.Background()
	// First create a client
	client, err := genai.NewClient(ctx, projectID, location)
	if err != nil {
		return "", fmt.Errorf("unable to create client: %v", err)
	}
	defer client.Close()
	return mm.SubmitToClient(ctx, client)
}
