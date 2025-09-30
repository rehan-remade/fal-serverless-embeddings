'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { trpc } from '@/lib/trpc/client';
import { toast } from 'sonner';
import { Input } from '@/components/ui/input';
import { Loader2, Upload, Search, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { PoweredByFalBadge } from '@/components/powered-by-fal-badge';
import { VideoGallery } from '@/components/video-gallery';

export default function VideoSearch() {
  const [searchQuery, setSearchQuery] = useState('');
  const [uploadedVideoUrl, setUploadedVideoUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  
  // Track if we're in search mode or random mode
  const [isInSearchMode, setIsInSearchMode] = useState(false);
  
  // Track loaded video IDs to avoid duplicates
  const loadedVideoIds = useRef<Set<string>>(new Set());
  
  // Track previous search values to avoid duplicate searches
  const previousSearchRef = useRef({ query: '', videoUrl: null as string | null });

  // Initial batch size and increment
  const INITIAL_LOAD = 50;
  const LOAD_MORE_COUNT = 30;

  // Mutation for initial random videos load
  const initialRandomMutation = trpc.embedding.getRandom.useMutation({
    onSuccess: (data: any) => {
      console.log('Initial random videos loaded:', data.embeddings.slice(0, 3)); // Debug: check first 3 items
      console.log('Video URLs:', data.embeddings.map((e: any) => ({ id: e.id, videoUrl: e.videoUrl, imageUrl: e.imageUrl })).slice(0, 5));
      
      loadedVideoIds.current.clear();
      data.embeddings.forEach((video: any) => loadedVideoIds.current.add(video.id));
      
      setSearchResults(data.embeddings.map((e: any) => ({
        ...e,
        distance: Math.random() * 0.5
      })));
      setIsInitialLoading(false);
    },
    onError: (error) => {
      console.error('Initial load error:', error);
      toast.error('Failed to load videos');
      setIsInitialLoading(false);
    }
  });

  // Load initial random videos on mount
  useEffect(() => {
    initialRandomMutation.mutate({ limit: INITIAL_LOAD });
  }, []); // Only run once on mount

  // Mutation for loading more random videos
  const loadMoreRandomMutation = trpc.embedding.getRandom.useMutation({
    onSuccess: (data: any) => {
      // Filter out already loaded videos
      const newVideos = data.embeddings.filter(
        (video: any) => !loadedVideoIds.current.has(video.id)
      );
      
      if (newVideos.length === 0) {
        setHasMore(false);
      } else {
        // Add new IDs to loaded set
        newVideos.forEach((video: any) => loadedVideoIds.current.add(video.id));
        
        // Append new videos with random distances
        setSearchResults(prev => [
          ...prev,
          ...newVideos.map((e: any) => ({
            ...e,
            distance: Math.random() * 0.5
          }))
        ]);
      }
      setIsLoadingMore(false);
    },
    onError: (error) => {
      console.error('Load more error:', error);
      setIsLoadingMore(false);
    }
  });

  // Upload mutation
  const uploadMutation = trpc.embedding.uploadVideo.useMutation({
    onSuccess: (data) => {
      setUploadedVideoUrl(data.url);
      toast.success('Video uploaded successfully!');
      setIsUploading(false);
    },
    onError: (error) => {
      console.error('Upload error:', error);
      toast.error('Failed to upload video: ' + error.message);
      setIsUploading(false);
    }
  });

  // TRPC mutation for search
  const searchMutation = trpc.embedding.search.useMutation({
    onSuccess: (data) => {
      // Clear loaded IDs when new search
      loadedVideoIds.current.clear();
      data.results.forEach(video => loadedVideoIds.current.add(video.id));
      
      setSearchResults(data.results);
      setIsSearching(false);
      setIsInSearchMode(true);
      // For now, disable loading more in search mode since backend doesn't support pagination
      setHasMore(false);
    },
    onError: (error) => {
      toast.error('Search failed: ' + error.message);
      setIsSearching(false);
    }
  });

  // Mutation for loading more search results
  const loadMoreSearchMutation = trpc.embedding.search.useMutation({
    onSuccess: (data) => {
      // Filter out already loaded videos
      const newVideos = data.results.filter(
        video => !loadedVideoIds.current.has(video.id)
      );
      
      if (newVideos.length === 0) {
        setHasMore(false);
      } else {
        // Add new IDs to loaded set
        newVideos.forEach(video => loadedVideoIds.current.add(video.id));
        
        // Append new videos, sorted by distance
        setSearchResults(prev => {
          const combined = [...prev, ...newVideos];
          // Sort by distance to maintain order
          return combined.sort((a, b) => (a.distance || 0) - (b.distance || 0));
        });
      }
      setIsLoadingMore(false);
    },
    onError: (error) => {
      console.error('Load more search error:', error);
      setIsLoadingMore(false);
    }
  });

  // Handle infinite scroll
  const handleScroll = useCallback(() => {
    if (isLoadingMore || !hasMore || isSearching || isInitialLoading) return;

    const scrollHeight = document.documentElement.scrollHeight;
    const scrollTop = document.documentElement.scrollTop;
    const clientHeight = document.documentElement.clientHeight;

    // Start loading when user is 80% down the page
    if (scrollTop + clientHeight >= scrollHeight * 0.8) {
      setIsLoadingMore(true);

      if (isInSearchMode && (searchQuery.trim() || uploadedVideoUrl)) {
        // Load more search results
        // Note: Since backend doesn't support offset, we'll request more and filter client-side
        loadMoreSearchMutation.mutate({
          text: searchQuery.trim() || undefined,
          videoUrl: uploadedVideoUrl || undefined,
          limit: searchResults.length + LOAD_MORE_COUNT
        });
      } else {
        // Load more random videos
        loadMoreRandomMutation.mutate({ limit: LOAD_MORE_COUNT });
      }
    }
  }, [isLoadingMore, hasMore, isSearching, isInitialLoading, isInSearchMode, searchQuery, uploadedVideoUrl, searchResults.length]);

  // Add scroll event listener
  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  // Debounce search as user types
  useEffect(() => {
    // If clearing search, reload initial random videos
    if (!searchQuery.trim() && !uploadedVideoUrl) {
      setIsInSearchMode(false);
      setHasMore(true);
      
      // Reload initial random videos
      initialRandomMutation.mutate({ limit: INITIAL_LOAD });
      return;
    }

    // Check if search values actually changed
    if (previousSearchRef.current.query === searchQuery && 
        previousSearchRef.current.videoUrl === uploadedVideoUrl) {
      return; // No change, don't search
    }

    const timer = setTimeout(() => {
      // Update previous values
      previousSearchRef.current = { query: searchQuery, videoUrl: uploadedVideoUrl };

      // Perform search
      setIsSearching(true);
      searchMutation.mutate({
        text: searchQuery.trim() || undefined,
        videoUrl: uploadedVideoUrl || undefined,
        limit: 50 // Start with more results for search
      });
    }, 800);

    return () => clearTimeout(timer);
  }, [searchQuery, uploadedVideoUrl]);

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    if (!file.type.startsWith('video/')) {
      toast.error('Please upload a video file');
      return;
    }

    setIsUploading(true);
    
    try {
      // Convert file to base64
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        const base64Data = base64.split(',')[1]; // Remove data:video/mp4;base64, prefix
        
        uploadMutation.mutate({
          fileBase64: base64Data,
          fileName: file.name,
          mimeType: file.type
        });
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error('Upload error:', error);
      toast.error('Failed to process video');
      setIsUploading(false);
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground relative">
      {/* Powered by Fal Badge */}
      <PoweredByFalBadge />

      {/* Main content area with drag and drop */}
      <div
        ref={dropZoneRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "min-h-screen px-4 py-8 pb-32", // Extra bottom padding for floating search bar
          isDragging && "relative"
        )}
      >
        {/* Drag overlay */}
        {isDragging && (
          <div className="fixed inset-0 z-40 bg-background/90 backdrop-blur-sm flex items-center justify-center">
            <div className="text-center">
              <Upload className="w-12 h-12 mx-auto mb-4 text-primary" />
              <p className="text-lg font-medium">Drop your video here</p>
            </div>
          </div>
        )}

        {/* Video Gallery */}
        <VideoGallery 
          results={searchResults} 
          isLoading={(isSearching || isInitialLoading) && searchResults.length === 0}
        />

        {/* Loading more indicator */}
        {isLoadingMore && (
          <div className="flex justify-center py-8">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        )}

        {/* Empty state */}
        {!isSearching && !isInitialLoading && searchResults.length === 0 && (searchQuery || uploadedVideoUrl) && (
          <div className="text-center py-20">
            <Search className="w-12 h-12 mx-auto mb-4 text-muted-foreground/50" />
            <p className="text-lg text-muted-foreground">No videos found</p>
            <p className="text-sm text-muted-foreground mt-2">
              Try adjusting your search query or upload a different video
            </p>
          </div>
        )}
      </div>

      {/* Floating search bar at bottom */}
      <div className="fixed bottom-6 left-4 right-4 z-50">
        <div className="max-w-3xl mx-auto">
          {/* Main search container */}
          <div className="relative">
            {/* Uploaded video preview - show above search bar */}
            {uploadedVideoUrl && (
              <div className="mb-3 flex justify-center">
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-black/80 backdrop-blur-xl rounded-full border border-white/20">
                  <div className="relative w-8 h-8 rounded-full overflow-hidden bg-secondary/20">
                    <video
                      src={uploadedVideoUrl}
                      className="w-full h-full object-cover"
                      muted
                    />
                  </div>
                  <span className="text-xs text-white">Video uploaded</span>
                  <button
                    onClick={() => setUploadedVideoUrl(null)}
                    className="p-1 hover:bg-white/10 rounded-full transition-colors"
                    title="Remove uploaded video"
                  >
                    <X className="w-3 h-3 text-white" />
                  </button>
                </div>
              </div>
            )}

            {/* Search input container */}
            <div className="relative bg-black/80 backdrop-blur-xl rounded-full border border-white/20 overflow-hidden">
              <div className="flex items-center">
                {/* Plus/Add button */}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploading}
                  className="flex-shrink-0 p-4 hover:bg-white/10 transition-colors disabled:opacity-50"
                >
                  {isUploading ? (
                    <Loader2 className="w-5 h-5 text-white animate-spin" />
                  ) : (
                    <div className="w-5 h-5 flex items-center justify-center">
                      <div className="w-4 h-0.5 bg-white rounded-full"></div>
                      <div className="w-0.5 h-4 bg-white rounded-full absolute"></div>
                    </div>
                  )}
                </button>

                {/* Search input */}
                <Input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Find me videos of..."
                  className={cn(
                    "flex-1 h-14 bg-transparent border-none text-white placeholder:text-white/60",
                    "focus:ring-0 focus:border-none focus-visible:ring-0 focus-visible:ring-offset-0",
                    "text-base px-0"
                  )}
                />

                {/* Right side icons */}
                <div className="flex items-center gap-2 pr-4">

                  {/* Search/Submit button */}
                  <button 
                    className="p-2 bg-white/20 hover:bg-white/30 rounded-full transition-colors"
                    onClick={() => {
                      if (searchQuery.trim() || uploadedVideoUrl) {
                        // Trigger search manually if needed
                      }
                    }}
                  >
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </button>
                </div>
              </div>
            </div>

            {/* File input (hidden) */}
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFileUpload(file);
              }}
              className="hidden"
            />
          </div>
        </div>
      </div>

      {/* Add CSS for animations */}
      <style jsx>{`
        @keyframes fadeInUp {
          to {
            opacity: 1;
            transform: translateY(0);
          }
          from {
            opacity: 0;
            transform: translateY(20px);
          }
        }
      `}</style>
    </div>
  );
}