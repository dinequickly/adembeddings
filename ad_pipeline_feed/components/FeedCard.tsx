'use client';

import { useState } from 'react';
import Image from 'next/image';
import type { ImageGroup } from '@/lib/types';

interface FeedCardProps {
  imageGroup: ImageGroup;
}

export default function FeedCard({ imageGroup }: FeedCardProps) {
  const [selectedView, setSelectedView] = useState<'original' | string>('original');

  const currentImageUrl =
    selectedView === 'original'
      ? imageGroup.original_url
      : imageGroup.variants.find((v) => v.brand === selectedView)?.image_url ||
        imageGroup.original_url;

  const hasVariants = imageGroup.variants.length > 0;

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="flex">
        {/* Left sidebar with buttons - only show if there are variants */}
        {hasVariants && (
          <div className="w-24 bg-gray-50 p-3 flex flex-col gap-2">
            <button
              onClick={() => setSelectedView('original')}
              className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                selectedView === 'original'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              Original
            </button>
            {imageGroup.variants.map((variant) => (
              <button
                key={variant.brand}
                onClick={() => setSelectedView(variant.brand)}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  selectedView === variant.brand
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-100'
                }`}
              >
                {variant.brand}
              </button>
            ))}
          </div>
        )}

        {/* Main image area */}
        <div className="flex-1">
          <div className="relative aspect-square bg-gray-100">
            <Image
              src={currentImageUrl}
              alt={`${selectedView === 'original' ? 'Original' : selectedView} - ${imageGroup.image_id}`}
              fill
              className="object-cover"
              sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
              priority={false}
            />
          </div>
          <div className="p-4">
            <p className="text-gray-500 text-sm truncate" title={imageGroup.image_id}>
              {imageGroup.image_id}
            </p>
            {!hasVariants && (
              <p className="text-orange-500 text-xs mt-1 font-medium">
                No variants yet - generate via Streamlit
              </p>
            )}
            <p className="text-gray-400 text-xs mt-1">
              {new Date(imageGroup.timestamp).toLocaleDateString()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
