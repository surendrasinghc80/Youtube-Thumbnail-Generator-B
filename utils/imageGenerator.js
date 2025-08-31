import { GoogleGenAI } from "@google/genai";
import OpenAI from "openai";
import mime from "mime";

class ImageGenerator {
  constructor() {
    this.ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
    });

    // Initialize OpenAI for image generation as fallback
    this.openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    });

    this.useOpenAI = true; // Use OpenAI DALL-E for now since Gemini image models have access issues
  }

  /**
   * Generates multiple images from a single prompt
   * @param {string} prompt - Text prompt for image generation
   * @param {number} count - Number of images to generate (default: 4)
   * @returns {Promise<Buffer[]>} - Array of image buffers
   */
  async generateImages(prompt, count = 4) {
    if (!process.env.GEMINI_API_KEY) {
      throw new Error("GEMINI_API_KEY environment variable is not set");
    }

    console.log(`Generating ${count} images for prompt: "${prompt}"`);

    const imageBuffers = [];
    const generatePromises = [];

    // Generate multiple images concurrently
    for (let i = 0; i < count; i++) {
      generatePromises.push(this.generateSingleImage(prompt, i + 1));
    }

    try {
      const results = await Promise.all(generatePromises);

      // Filter out any failed generations and collect buffers
      for (const result of results) {
        if (result && result.length > 0) {
          imageBuffers.push(...result);
        }
      }

      console.log(`Successfully generated ${imageBuffers.length} images`);
      return imageBuffers;
    } catch (error) {
      console.error("Error generating images:", error);
      throw error;
    }
  }

  /**
   * Generates a single image from a prompt
   * @param {string} prompt - Text prompt for image generation
   * @param {number} index - Index for logging purposes
   * @returns {Promise<Buffer[]>} - Array containing image buffer(s)
   */
  async generateSingleImage(prompt, index) {
    try {
      console.log(
        `Generating image ${index} with OpenAI DALL-E and prompt: "${prompt}"`
      );

      if (this.useOpenAI) {
        const response = await this.openai.images.generate({
          model: "dall-e-3",
          prompt: prompt,
          n: 1,
          size: "1024x1024",
          quality: "standard",
          response_format: "b64_json",
        });

        const imageBuffers = [];

        if (response.data && response.data.length > 0) {
          for (const imageData of response.data) {
            if (imageData.b64_json) {
              const buffer = Buffer.from(imageData.b64_json, "base64");
              imageBuffers.push(buffer);
              console.log(
                `Image ${index} generated successfully with DALL-E - buffer size: ${buffer.length} bytes`
              );
            }
          }
        }

        console.log(
          `Final result for image ${index}: ${imageBuffers.length} buffers generated`
        );
        return imageBuffers;
      }
    } catch (error) {
      console.error(`Error generating image ${index} with OpenAI:`, {
        message: error.message,
        status: error.status,
        code: error.code,
      });
    }

    console.log(`Image generation failed for image ${index}`);
    return [];
  }

  /**
   * Generates multiple images from an input image and prompt
   * @param {Buffer} imageBuffer - Input image buffer
   * @param {string} prompt - Text prompt for image generation
   * @param {number} count - Number of images to generate (default: 4)
   * @returns {Promise<Buffer[]>} - Array of image buffers
   */
  async generateImagesFromImage(imageBuffer, prompt, count = 4) {
    if (!process.env.GEMINI_API_KEY) {
      throw new Error("GEMINI_API_KEY environment variable is not set");
    }

    console.log(
      `Generating ${count} images from input image with prompt: "${prompt}"`
    );

    const imageBuffers = [];
    const generatePromises = [];

    // Generate multiple images concurrently
    for (let i = 0; i < count; i++) {
      generatePromises.push(
        this.generateSingleImageFromImage(imageBuffer, prompt, i + 1)
      );
    }

    try {
      const results = await Promise.all(generatePromises);

      // Filter out any failed generations and collect buffers
      for (const result of results) {
        if (result && result.length > 0) {
          imageBuffers.push(...result);
        }
      }

      console.log(
        `Successfully generated ${imageBuffers.length} images from input image`
      );
      return imageBuffers;
    } catch (error) {
      console.error("Error generating images from input image:", error);
      throw error;
    }
  }

  /**
   * Generates a single image from an input image and prompt
   * @param {Buffer} imageBuffer - Input image buffer
   * @param {string} prompt - Text prompt for image generation
   * @param {number} index - Index for logging purposes
   * @returns {Promise<Buffer[]>} - Array containing image buffer(s)
   */
  async generateSingleImageFromImage(imageBuffer, prompt, index) {
    const base64Image = imageBuffer.toString("base64");
    const mimeType = this.detectMimeType(imageBuffer);

    // Try Gemini 2.5 Flash Image Preview first for superior image-to-image quality
    try {
      console.log(
        `Generating image ${index} from input image with Gemini 2.5 Flash Image Preview...`
      );

      const enhancedPrompt = `Transform and enhance this image: ${prompt}. Create a professional YouTube thumbnail style image with high quality, vibrant colors, and eye-catching design. Maintain key visual elements from the reference image while incorporating the requested changes.`;

      // Add delay to avoid rate limits
      if (index > 1) {
        await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second delay between requests
      }

      // Use exact structure from official Gemini docs
      const promptArray = [
        { 
          text: enhancedPrompt 
        },
        {
          inlineData: {
            mimeType: mimeType,
            data: base64Image,
          },
        },
      ];

      const response = await this.ai.models.generateContent({
        model: "gemini-2.5-flash-image-preview",
        contents: promptArray,
      });

      const imageBuffers = [];

      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.text) {
            console.log(`AI Response for image ${index}:`, part.text);
          } else if (part.inlineData) {
            const buffer = Buffer.from(part.inlineData.data, "base64");
            imageBuffers.push(buffer);
            console.log(
              `Image ${index} generated successfully with Gemini 2.5 Flash Image Preview - buffer size: ${buffer.length} bytes`
            );
          }
        }
      }

      if (imageBuffers.length > 0) {
        return imageBuffers;
      }
    } catch (error) {
      console.error(`Error generating image ${index} with Gemini:`, {
        message: error.message,
        status: error.status,
        details: error.details,
      });

      // If it's a rate limit error (429), wait and retry once
      if (error.status === 429) {
        console.log(`Rate limited, waiting 5 seconds before retry for image ${index}...`);
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        try {
          const retryPromptArray = [
            { 
              text: enhancedPrompt 
            },
            {
              inlineData: {
                mimeType: mimeType,
                data: base64Image,
              },
            },
          ];

          const retryResponse = await this.ai.models.generateContent({
            model: "gemini-2.5-flash-image-preview",
            contents: retryPromptArray,
          });

          const retryImageBuffers = [];
          if (retryResponse.candidates?.[0]?.content?.parts) {
            for (const part of retryResponse.candidates[0].content.parts) {
              if (part.inlineData) {
                const buffer = Buffer.from(part.inlineData.data, "base64");
                retryImageBuffers.push(buffer);
                console.log(
                  `Image ${index} generated successfully with Gemini retry - buffer size: ${buffer.length} bytes`
                );
              }
            }
          }
          
          if (retryImageBuffers.length > 0) {
            return retryImageBuffers;
          }
        } catch (retryError) {
          console.error(`Retry also failed for image ${index}:`, retryError.message);
        }
      }

      // If Gemini fails, fallback to OpenAI DALL-E
      console.log(`Falling back to OpenAI DALL-E for image ${index}...`);
    }

    // Fallback to OpenAI DALL-E
    try {
      console.log(
        `Generating image ${index} from input image with OpenAI DALL-E (fallback)...`
      );

      const enhancedPrompt = `Transform and enhance this concept: ${prompt}. Create a professional YouTube thumbnail style image with high quality, vibrant colors, and eye-catching design.`;

      const response = await this.openai.images.generate({
        model: "dall-e-3",
        prompt: enhancedPrompt,
        n: 1,
        size: "1792x1024", // YouTube thumbnail aspect ratio (16:9)
        quality: "standard",
        response_format: "b64_json",
      });

      const imageBuffers = [];

      if (response.data && response.data.length > 0) {
        for (const imageData of response.data) {
          if (imageData.b64_json) {
            const buffer = Buffer.from(imageData.b64_json, "base64");
            imageBuffers.push(buffer);
            console.log(
              `Image ${index} generated successfully with DALL-E fallback - buffer size: ${buffer.length} bytes`
            );
          }
        }
      }

      return imageBuffers;
    } catch (error) {
      console.error(`Error generating image ${index} with OpenAI fallback:`, {
        message: error.message,
        status: error.status,
        code: error.code,
      });
    }

    console.log(
      `All image-to-image generation methods failed for image ${index}`
    );
    return [];
  }

  /**
   * Detects MIME type from image buffer
   * @param {Buffer} buffer - Image buffer
   * @returns {string} - MIME type
   */
  detectMimeType(buffer) {
    const signatures = {
      "image/jpeg": [0xff, 0xd8, 0xff],
      "image/png": [0x89, 0x50, 0x4e, 0x47],
      "image/gif": [0x47, 0x49, 0x46],
      "image/webp": [0x52, 0x49, 0x46, 0x46],
    };

    for (const [mimeType, signature] of Object.entries(signatures)) {
      if (signature.every((byte, index) => buffer[index] === byte)) {
        return mimeType;
      }
    }

    // Default to JPEG if unable to detect
    return "image/jpeg";
  }

  /**
   * Validates if the image generator is properly configured
   * @returns {boolean} - True if API key is available
   */
  isConfigured() {
    return !!process.env.GEMINI_API_KEY;
  }
}

export default ImageGenerator;
