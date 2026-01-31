export interface Variant {
  id: string;              // "image_id_brand"
  image_id: string;        // "Gemini_Generated_Image_uhwt5guhwt5guhwt"
  brand: string;           // "Coke" | "Pepsi"
  image_url: string;       // Supabase URL or "/api/images/..."
  source: 'n8n' | 'local'; // Where image came from
  timestamp?: number;      // Optional: for sorting
}

export interface N8NResult {
  image_url: string;
  timestamp: number;
  image_id: string;
  brand: string;
}

export interface N8NResults {
  [key: string]: N8NResult;
}
