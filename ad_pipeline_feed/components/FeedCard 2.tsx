import Image from 'next/image';
import type { Variant } from '@/lib/types';

interface FeedCardProps {
  variant: Variant;
}

export default function FeedCard({ variant }: FeedCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow duration-300">
      <div className="relative aspect-square bg-gray-100">
        <Image
          src={variant.image_url}
          alt={`${variant.brand} variant for ${variant.image_id}`}
          fill
          className="object-cover rounded-t-lg"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw"
          priority={false}
        />
      </div>
      <div className="p-4">
        <h3 className="font-semibold text-lg text-gray-900">{variant.brand}</h3>
        <p className="text-gray-500 text-sm truncate" title={variant.image_id}>
          {variant.image_id}
        </p>
        {variant.timestamp && (
          <p className="text-gray-400 text-xs mt-1">
            {new Date(variant.timestamp).toLocaleDateString()}
          </p>
        )}
      </div>
    </div>
  );
}
