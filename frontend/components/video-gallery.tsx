'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import { Play } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Skeleton } from '@/components/ui/skeleton';

interface VideoResult {
  id: string;
  videoUrl: string | null;
  imageUrl: string | null;
  text: string | null;
  createdAt: Date;
  distance?: number;
}

interface VideoGalleryProps {
  results: VideoResult[];
  isLoading?: boolean;
}

// Helper to determine aspect ratio class from dimensions
const getAspectRatioClass = (width: number, height: number): string => {
  const ratio = width / height;
  
  if (ratio > 0.9 && ratio < 1.1) return 'aspect-square';
  if (ratio > 1.7 && ratio < 1.8) return 'aspect-video';
  if (ratio > 0.5 && ratio < 0.6) return 'aspect-[9/16]';
  if (ratio > 1.3 && ratio < 1.4) return 'aspect-[4/3]';
  if (ratio > 0.7 && ratio < 0.8) return 'aspect-[3/4]';
  if (ratio > 2.3 && ratio < 2.4) return 'aspect-[21/9]';
  
  // For other ratios, calculate a simplified ratio
  const gcd = (a: number, b: number): number => b === 0 ? a : gcd(b, a % b);
  const divisor = gcd(width, height);
  const simplifiedWidth = Math.round(width / divisor);
  const simplifiedHeight = Math.round(height / divisor);
  
  return `aspect-[${simplifiedWidth}/${simplifiedHeight}]`;
};

export const VideoGallery: React.FC<VideoGalleryProps> = ({ results, isLoading = false }) => {
  const [loadedStates, setLoadedStates] = useState<Record<string, boolean>>({});
  const [aspectClasses, setAspectClasses] = useState<Record<string, string>>({});
  const [playingVideos, setPlayingVideos] = useState<Set<string>>(new Set());
  const videoRefs = useRef<Record<string, HTMLVideoElement>>({});
  
  // Get column count based on screen size
  const getColumnCount = () => {
    if (typeof window === 'undefined') return 3;
    const width = window.innerWidth;
    if (width < 640) return 1;
    if (width < 1024) return 2;
    if (width < 1280) return 3;
    return 4;
  };

  // Distribute items into columns
  const columns = useMemo(() => {
    const columnCount = getColumnCount();
    const cols: VideoResult[][] = Array(columnCount).fill(null).map(() => []);
    
    results.forEach((item, index) => {
      const columnIndex = index % columnCount;
      cols[columnIndex].push(item);
    });
    
    return cols;
  }, [results]);

  const handleVideoMetadata = (e: React.SyntheticEvent<HTMLVideoElement>, id: string) => {
    const video = e.currentTarget;
    videoRefs.current[id] = video; // Store ref
    
    if (video.videoWidth && video.videoHeight) {
      const aspectClass = getAspectRatioClass(video.videoWidth, video.videoHeight);
      setAspectClasses(prev => ({ ...prev, [id]: aspectClass }));
    }
    setLoadedStates(prev => ({ ...prev, [id]: true }));
  };

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>, id: string) => {
    const img = e.currentTarget;
    if (img.naturalWidth && img.naturalHeight) {
      const aspectClass = getAspectRatioClass(img.naturalWidth, img.naturalHeight);
      setAspectClasses(prev => ({ ...prev, [id]: aspectClass }));
    }
    setLoadedStates(prev => ({ ...prev, [id]: true }));
  };

  const handleMouseEnter = (id: string) => {
    const video = videoRefs.current[id];
    if (video) {
      video.play().catch((err) => {
        console.log('Video play failed:', err);
      });
      setPlayingVideos(prev => new Set(prev).add(id));
    }
  };

  const handleMouseLeave = (id: string) => {
    const video = videoRefs.current[id];
    if (video) {
      video.pause();
      video.currentTime = 0;
      setPlayingVideos(prev => {
        const newSet = new Set(prev);
        newSet.delete(id);
        return newSet;
      });
    }
  };

  // Clean up refs when results change
  useEffect(() => {
    return () => {
      videoRefs.current = {};
    };
  }, [results]);

  if (isLoading && results.length === 0) {
    return <LoadingGrid />;
  }

  return (
    <div className="container mx-auto">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {columns.map((column, columnIndex) => (
          <div key={`column-${columnIndex}`} className="flex flex-col gap-4">
            {column.map((result) => {
              const aspectClass = aspectClasses[result.id] || 'aspect-video';
              const isLoaded = loadedStates[result.id];
              const isPlaying = playingVideos.has(result.id);

              return (
                <div
                  key={result.id}
                  className={cn(
                    "group relative overflow-hidden rounded-xl",
                    "hover:scale-[1.02] transition-transform duration-300",
                    "cursor-pointer bg-secondary/20"
                  )}
                  style={{
                    animation: 'fadeInUp 0.5s ease-out forwards',
                    animationDelay: `${columnIndex * 100}ms`
                  }}
                  onMouseEnter={() => result.videoUrl && handleMouseEnter(result.id)}
                  onMouseLeave={() => result.videoUrl && handleMouseLeave(result.id)}
                >
                  <div className={cn("relative w-full", aspectClass)}>
                    {result.videoUrl ? (
                      <>
                        <video
                          ref={(el) => {
                            if (el) videoRefs.current[result.id] = el;
                          }}
                          src={result.videoUrl}
                          className={cn(
                            "w-full h-full object-cover",
                            "transition-opacity duration-300",
                            isLoaded ? "opacity-100" : "opacity-0"
                          )}
                          muted
                          loop
                          playsInline
                          preload="metadata"
                          onLoadedMetadata={(e) => handleVideoMetadata(e, result.id)}
                        />
                        
                        {/* Loading spinner */}
                        {!isLoaded && (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          </div>
                        )}
                        
                        {/* Play overlay */}
                        {isLoaded && !isPlaying && (
                          <div className="absolute inset-0 bg-black/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none">
                            <div className="bg-white/90 rounded-full p-3">
                              <Play className="w-6 h-6 text-black fill-black" />
                            </div>
                          </div>
                        )}
                      </>
                    ) : result.imageUrl ? (
                      <>
                        <img
                          src={result.imageUrl}
                          alt=""
                          className={cn(
                            "w-full h-full object-cover",
                            "transition-opacity duration-300",
                            isLoaded ? "opacity-100" : "opacity-0"
                          )}
                          onLoad={(e) => handleImageLoad(e, result.id)}
                        />
                        
                        {/* Loading skeleton */}
                        {!isLoaded && (
                          <div className="absolute inset-0">
                            <Skeleton className="w-full h-full" />
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-secondary/50">
                        <Play className="w-12 h-12 text-muted-foreground/50" />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};

// Loading grid component
const LoadingGrid: React.FC = () => {
  const skeletonClasses = [
    'aspect-video',
    'aspect-square',
    'aspect-[9/16]',
    'aspect-[4/3]',
    'aspect-[3/4]',
  ];

  return (
    <div className="container mx-auto">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {[0, 1, 2, 3].map((columnIndex) => (
          <div key={`column-${columnIndex}`} className="flex flex-col gap-4">
            {[0, 1, 2].map((itemIndex) => {
              const aspectClass = skeletonClasses[(columnIndex + itemIndex) % skeletonClasses.length];
              return (
                <div
                  key={`skeleton-${columnIndex}-${itemIndex}`}
                  className="rounded-xl overflow-hidden"
                  style={{
                    animation: 'fadeInUp 0.5s ease-out forwards',
                    animationDelay: `${(columnIndex * 100) + (itemIndex * 50)}ms`,
                    opacity: 0
                  }}
                >
                  <Skeleton className={cn("w-full", aspectClass)} />
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
};
