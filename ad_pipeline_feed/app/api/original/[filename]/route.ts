import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

const AD_PIPELINE_DIR = path.join(process.cwd(), '..', 'ad_pipeline');
const RAW_DIR = path.join(AD_PIPELINE_DIR, 'data', 'images', 'raw');

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  try {
    const { filename } = await params;

    if (!filename) {
      return new Response('Bad Request', { status: 400 });
    }

    // Construct safe path
    const safePath = path.join(RAW_DIR, filename);

    // Security check: ensure path is within RAW_DIR
    const normalizedSafePath = path.normalize(safePath);
    const normalizedRawDir = path.normalize(RAW_DIR);

    if (!normalizedSafePath.startsWith(normalizedRawDir)) {
      console.error('Path traversal attempt:', filename);
      return new Response('Forbidden', { status: 403 });
    }

    // Check if file exists
    if (!fs.existsSync(safePath)) {
      return new Response('Not Found', { status: 404 });
    }

    // Read file
    const fileBuffer = fs.readFileSync(safePath);

    // Determine content type
    const ext = path.extname(safePath).toLowerCase();
    const contentType =
      ext === '.png'
        ? 'image/png'
        : ext === '.jpg' || ext === '.jpeg'
        ? 'image/jpeg'
        : 'application/octet-stream';

    // Return image with proper headers
    return new Response(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=31536000, immutable',
      },
    });
  } catch (error) {
    console.error('Error serving original image:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}
