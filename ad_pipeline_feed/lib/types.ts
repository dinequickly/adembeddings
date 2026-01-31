export interface N8NResult {
  image_url: string;
  timestamp: number;
  image_id: string;
  brand: string;
}

export interface N8NResults {
  [key: string]: N8NResult;
}

export interface ImageVariant {
  brand: string;
  image_url: string;
  timestamp: number;
}

export interface ImageGroup {
  image_id: string;
  original_url: string;
  variants: ImageVariant[];
  timestamp: number;
}
