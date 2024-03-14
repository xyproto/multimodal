package main

import (
	"fmt"
	"log"

	"github.com/xyproto/env/v2"
	"github.com/xyproto/multimodal"
	"github.com/xyproto/wordwrap"
)

func main() {
	mm := multimodal.New("gemini-1.0-pro-vision", 0.4)

	// Build a prompt

	mm.AddImage("frog.png")
	mm.AddURI("gs://generativeai-downloads/images/scones.jpg")
	mm.AddText("describe what is common for these two images")

	location := env.Str("GCP_LOCATION", "us-central1")
	projectID := env.StrAlt("GCP_PROJECT", "GCLOUD_PROJECT", "")
	if projectID == "" {
		log.Fatalln("Please set GCP_PROJECT or GCLOUD_PROJECT to your Google Cloud project ID, for a multimodal Vertex AI model")
	}

	response, err := mm.Submit(projectID, location)
	if err != nil {
		log.Fatalln(err)
	}

	if lines, err := wordwrap.WordWrap(response, 79); err == nil { // success
		for _, line := range lines {
			fmt.Println(line)
		}
		return
	}

	fmt.Println(response)
}
