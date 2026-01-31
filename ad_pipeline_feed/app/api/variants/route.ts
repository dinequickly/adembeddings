import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import type { N8NResults } from '@/lib/types';

const AD_PIPELINE_DIR = path.join(process.cwd(), '..', 'ad_pipeline');
const N8N_RESULTS_PATH = path.join(AD_PIPELINE_DIR, 'data', 'n8n_results.json');
const RAW_DIR = path.join(AD_PIPELINE_DIR, 'data', 'images', 'raw');
const MASK_DIR = path.join(AD_PIPELINE_DIR, 'data', 'images', 'masks');

interface ImageGroup {
  image_id: string;
  original_url: string;
  variants: {
    brand: string;
    image_url: string;
    timestamp: number;
  }[];
  timestamp: number;
}

export async function GET() {
  try {
    const imageGroups: { [key: string]: ImageGroup } = {};

    // First, scan masks directory to find all images with masks
    if (fs.existsSync(MASK_DIR)) {
      const maskFiles = fs.readdirSync(MASK_DIR);

      maskFiles.forEach((file) => {
        // Look for mask PNG files
        if (file.endsWith('_mask.png')) {
          const imageId = file.replace('_mask.png', '');

          // Check if original image exists
          const possibleExtensions = ['.png', '.jpg', '.jpeg'];
          let originalExists = false;

          for (const ext of possibleExtensions) {
            const originalPath = path.join(RAW_DIR, `${imageId}${ext}`);
            if (fs.existsSync(originalPath)) {
              originalExists = true;

              // Create image group if not already exists
              if (!imageGroups[imageId]) {
                imageGroups[imageId] = {
                  image_id: imageId,
                  original_url: `/api/original/${imageId}${ext}`,
                  variants: [],
                  timestamp: fs.statSync(originalPath).mtimeMs,
                };
              }
              break;
            }
          }
        }
      });
    }

    // Load N8N results and add variants to existing image groups
    if (fs.existsSync(N8N_RESULTS_PATH)) {
      try {
        const n8nData = fs.readFileSync(N8N_RESULTS_PATH, 'utf-8');
        const n8nResults: N8NResults = JSON.parse(n8nData);

        // Group variants by image_id
        Object.entries(n8nResults).forEach(([key, result]) => {
          const { image_id, brand, image_url, timestamp } = result;

          // If image group doesn't exist yet, create it
          if (!imageGroups[image_id]) {
            imageGroups[image_id] = {
              image_id,
              original_url: `/api/original/${image_id}.png`,
              variants: [],
              timestamp: timestamp,
            };
          }

          imageGroups[image_id].variants.push({
            brand,
            image_url,
            timestamp,
          });

          // Update timestamp to latest
          if (timestamp > imageGroups[image_id].timestamp) {
            imageGroups[image_id].timestamp = timestamp;
          }
        });
      } catch (error) {
        console.error('Error reading N8N results:', error);
      }
    }

    // Convert to array and sort by timestamp
    const result = Object.values(imageGroups).sort(
      (a, b) => b.timestamp - a.timestamp
    );

    return NextResponse.json(result);
  } catch (error) {
    console.error('Error in /api/variants:', error);
    return NextResponse.json(
      { error: 'Failed to fetch variants' },
      { status: 500 }
    );
  }
}
