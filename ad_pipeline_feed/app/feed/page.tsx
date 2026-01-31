'use client';

import { useEffect, useState } from 'react';
import FeedCard from '@/components/FeedCard';
import type { ImageGroup } from '@/lib/types';

export default function FeedPage() {
  const [imageGroups, setImageGroups] = useState<ImageGroup[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchVariants() {
      try {
        const response = await fetch('/api/variants');
        if (!response.ok) {
          throw new Error('Failed to fetch variants');
        }
        const data = await response.json();
        setImageGroups(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }

    fetchVariants();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">Ad Variants Feed</h1>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="bg-white rounded-lg shadow-md animate-pulse">
                <div className="flex">
                  <div className="w-24 bg-gray-100"></div>
                  <div className="flex-1">
                    <div className="aspect-square bg-gray-200"></div>
                    <div className="p-4 space-y-2">
                      <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-md p-8 max-w-md">
          <h2 className="text-xl font-semibold text-red-600 mb-2">Error</h2>
          <p className="text-gray-700">{error}</p>
        </div>
      </div>
    );
  }

  if (imageGroups.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-lg shadow-md p-8 max-w-md text-center">
          <h2 className="text-xl font-semibold text-gray-900 mb-2">No Variants Yet</h2>
          <p className="text-gray-600">
            Generate some variants using the Streamlit app to see them here.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Ad Variants Feed</h1>
          <p className="text-gray-600 mt-2">{imageGroups.length} images with variants</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {imageGroups.map((imageGroup) => (
            <FeedCard key={imageGroup.image_id} imageGroup={imageGroup} />
          ))}
        </div>
      </div>
    </div>
  );
}
