package multimodal

import (
	"context"
	"errors"
	"fmt"
	"log"
	"mime"
	"os"
	"path/filepath"
	"strings"

	"cloud.google.com/go/vertexai/genai"
)

type MultiModal struct {
	modelName   string
	temperature float32
	parts       []genai.Part
	trim        bool
	verbose     bool
}

func New(modelName string, temperature float32) *MultiModal {
	parts := make([]genai.Part, 0)
	return &MultiModal{modelName, 0.4, parts, true, false}
}

func (mm *MultiModal) SetVerbose(verbose bool) {
	mm.verbose = verbose
}

func (mm *MultiModal) SetTrim(trim bool) {
	mm.trim = trim
}

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

func (mm *MultiModal) MustAddImage(filename string) {
	if err := mm.AddImage(filename); err != nil {
		log.Fatalln(err)
	}
}

// AddURI takes an URI and adds a genai.Part (a genai.FileData).
// Example URI: "gs://generativeai-downloads/images/scones.jpg"
func (mm *MultiModal) AddURI(URI string) {
	mm.parts = append(mm.parts, genai.FileData{
		MIMEType: mime.TypeByExtension(filepath.Ext(URI)),
		FileURI:  URI,
	})
}

// AddURIWithMIME takes an URI and adds a genai.Part (a genai.FileData).
// Also takes a MIME type.
// Example URI: "gs://generativeai-downloads/images/scones.jpg"
func (mm *MultiModal) AddURIWithMIME(URI, MIME string) {
	mm.parts = append(mm.parts, genai.FileData{
		MIMEType: MIME,
		FileURI:  URI,
	})
}

func (mm *MultiModal) AddText(prompt string) {
	mm.parts = append(mm.parts, genai.Text(prompt))
}

func (mm *MultiModal) Submit(projectID, location string) (string, error) {
	ctx := context.Background()
	// First create a client
	client, err := genai.NewClient(ctx, projectID, location)
	if err != nil {
		return "", fmt.Errorf("unable to create client: %v", err)
	}
	defer client.Close()
	// Then configure the model
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
