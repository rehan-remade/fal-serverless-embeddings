'use client';

import { useState, useRef, useEffect } from 'react';
import { Play, Video } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Skeleton } from '@/components/ui/skeleton';

interface VideoItemProps {
  result: {
    id: string;
    videoUrl: string | null;
    imageUrl: string | null;
    text: string | null;
    createdAt: Date;
    distance?: number;
  };
  index: number;
}

export const VideoItem: React.FC<VideoItemProps> = ({ result, index }) => {
  const [aspectRatio, setAspectRatio] = useState<number>(16/9); // Default aspect ratio
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (result.videoUrl && videoRef.current) {
      const video = videoRef.current;
      
      const handleLoadedMetadata = () => {
        if (video.videoWidth && video.videoHeight) {
          const ratio = video.videoWidth / video.videoHeight;
          setAspectRatio(ratio);
        }
        setIsLoading(false);
      };

      const handleError = () => {
        setHasError(true);
        setIsLoading(false);
      };

      const handlePlay = () => setIsPlaying(true);
      const handlePause = () => setIsPlaying(false);

      video.addEventListener('loadedmetadata', handleLoadedMetadata);
      video.addEventListener('error', handleError);
      video.addEventListener('play', handlePlay);
      video.addEventListener('pause', handlePause);

      return () => {
        video.removeEventListener('loadedmetadata', handleLoadedMetadata);
        video.removeEventListener('error', handleError);
        video.removeEventListener('play', handlePlay);
        video.removeEventListener('pause', handlePause);
      };
    } else if (result.imageUrl) {
      setIsLoading(false);
    } else {
      setIsLoading(false);
    }
  }, [result.videoUrl, result.imageUrl]);

  const handleMouseEnter = () => {
    if (videoRef.current && !hasError) {
      videoRef.current.play().catch(() => {
        // Handle play promise rejection silently
      });
    }
  };

  const handleMouseLeave = () => {
    if (videoRef.current && !hasError) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  };

  if (isLoading) {
    return (
      <div
        className="break-inside-avoid mb-4"
        style={{
          animationDelay: `${index * 50}ms`,
          animation: 'fadeInUp 0.5s ease-out forwards',
          opacity: 0
        }}
      >
        <div className="rounded-xl overflow-hidden">
          <Skeleton 
            className="w-full"
            style={{ 
              aspectRatio: aspectRatio.toString(),
              minHeight: '200px'
            }}
          />
        </div>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "break-inside-avoid mb-4",
        "group relative overflow-hidden rounded-xl",
        "hover:scale-[1.02] transition-all duration-300",
        "cursor-pointer"
      )}
      style={{
        animationDelay: `${index * 50}ms`,
        animation: 'fadeInUp 0.5s ease-out forwards',
        opacity: 0
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Video/Image content */}
      <div 
        className="relative bg-secondary/20 rounded-xl overflow-hidden"
        style={{ 
          aspectRatio: aspectRatio.toString(),
          minHeight: '200px'
        }}
      >
        {result.videoUrl && !hasError ? (
          <video
            ref={videoRef}
            src={result.videoUrl}
            className="w-full h-full object-cover"
            muted
            loop
            preload="metadata"
            playsInline
          />
        ) : result.imageUrl ? (
          <img
            src={result.imageUrl}
            alt=""
            className="w-full h-full object-cover"
            onLoad={(e) => {
              const img = e.currentTarget;
              if (img.naturalWidth && img.naturalHeight) {
                const ratio = img.naturalWidth / img.naturalHeight;
                setAspectRatio(ratio);
              }
            }}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <Video className="w-12 h-12 text-muted-foreground/50" />
          </div>
        )}
        
        {/* Play overlay - only show when not playing */}
        {result.videoUrl && !isPlaying && (
          <div className="absolute inset-0 bg-black/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <div className="bg-white/90 rounded-full p-3">
              <Play className="w-6 h-6 text-black fill-black" />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
