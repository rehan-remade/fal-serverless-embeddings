'use client';

import { useState, useEffect, useCallback } from 'react';
import { trpc } from '@/lib/trpc/client';
import { toast } from 'sonner';
import { Loader2, Trash2, Video, Image as ImageIcon, FileText, Play, Pause, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { PoweredByFalBadge } from '@/components/powered-by-fal-badge';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Navigation } from '@/components/navigation';

const ITEMS_PER_PAGE = 100; // Maximum allowed by API

export default function AdminPage() {
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [playingVideos, setPlayingVideos] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(1);

  // Calculate offset
  const offset = (currentPage - 1) * ITEMS_PER_PAGE;

  // Fetch embeddings with pagination
  const { data, isLoading, refetch } = trpc.embedding.list.useQuery(
    { 
      limit: ITEMS_PER_PAGE,
      offset: offset
    },
    {
      refetchOnWindowFocus: false,
      keepPreviousData: true, // Keep previous data while loading new page
    }
  );

  // Calculate pagination info
  const totalItems = data?.total || 0;
  const totalPages = Math.ceil(totalItems / ITEMS_PER_PAGE);
  const allEmbeddings = data?.embeddings || [];

  // Delete mutation
  const deleteMutation = trpc.embedding.delete.useMutation({
    onSuccess: () => {
      toast.success('Video deleted successfully');
      setShowDeleteDialog(false);
      setSelectedVideo(null);
      refetch();
    },
    onError: (error) => {
      toast.error('Failed to delete video: ' + error.message);
    },
    onSettled: () => {
      setDeletingId(null);
    }
  });

  const handleDelete = (id: string) => {
    setSelectedVideo(id);
    setShowDeleteDialog(true);
  };

  const confirmDelete = () => {
    if (selectedVideo) {
      setDeletingId(selectedVideo);
      deleteMutation.mutate({ id: selectedVideo });
    }
  };

  const toggleVideoPlay = (id: string, videoElement: HTMLVideoElement | null) => {
    if (!videoElement) return;

    if (playingVideos.has(id)) {
      videoElement.pause();
      setPlayingVideos(prev => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    } else {
      // Pause all other videos
      playingVideos.forEach(playingId => {
        if (playingId !== id) {
          const otherVideo = document.getElementById(`video-${playingId}`) as HTMLVideoElement;
          if (otherVideo) otherVideo.pause();
        }
      });

      videoElement.play().catch(err => {
        console.error('Failed to play video:', err);
        toast.error('Failed to play video');
      });
      
      setPlayingVideos(new Set([id]));
    }
  };

  const getMediaIcon = (item: any) => {
    if (item.videoUrl) return <Video className="w-4 h-4" />;
    if (item.imageUrl) return <ImageIcon className="w-4 h-4" />;
    return <FileText className="w-4 h-4" />;
  };

  const getMediaPreview = (item: any) => {
    if (item.videoUrl) {
      const isPlaying = playingVideos.has(item.id);
      return (
        <div className="relative w-full h-full group/video">
          <video
            id={`video-${item.id}`}
            src={item.videoUrl}
            className="w-full h-full object-cover"
            muted
            loop
            playsInline
            preload="metadata"
            onEnded={() => {
              setPlayingVideos(prev => {
                const next = new Set(prev);
                next.delete(item.id);
                return next;
              });
            }}
            onError={(e) => {
              console.error('Video error:', e);
            }}
          />
          {/* Play/Pause overlay */}
          <button
            onClick={(e) => {
              e.stopPropagation();
              const video = document.getElementById(`video-${item.id}`) as HTMLVideoElement;
              toggleVideoPlay(item.id, video);
            }}
            className="absolute inset-0 flex items-center justify-center bg-black/20 opacity-0 group-hover/video:opacity-100 transition-opacity"
          >
            <div className="p-3 bg-black/60 rounded-full backdrop-blur-sm">
              {isPlaying ? (
                <Pause className="w-6 h-6 text-white" />
              ) : (
                <Play className="w-6 h-6 text-white" />
              )}
            </div>
          </button>
        </div>
      );
    }
    if (item.imageUrl) {
      return (
        <img
          src={item.imageUrl}
          alt="Embedded content"
          className="w-full h-full object-cover"
          loading="lazy"
        />
      );
    }
    return (
      <div className="w-full h-full bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
        <FileText className="w-12 h-12 text-white/50" />
      </div>
    );
  };

  // Pagination handlers
  const goToPage = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
      window.scrollTo(0, 0);
    }
  };

  const getPageNumbers = () => {
    const pages = [];
    const maxVisible = 7;
    const halfVisible = Math.floor(maxVisible / 2);

    let start = Math.max(1, currentPage - halfVisible);
    let end = Math.min(totalPages, currentPage + halfVisible);

    if (currentPage <= halfVisible) {
      end = Math.min(totalPages, maxVisible);
    }
    if (currentPage + halfVisible >= totalPages) {
      start = Math.max(1, totalPages - maxVisible + 1);
    }

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    return pages;
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Navigation */}
      <Navigation />

      {/* Powered by Fal Badge */}
      <PoweredByFalBadge />

      {/* Header */}
      <div className="sticky top-0 z-40 bg-background/80 backdrop-blur-xl border-b border-border">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Admin Dashboard
              </h1>
              <p className="text-muted-foreground mt-1">
                Manage your video embeddings database
              </p>
            </div>
            <div className="text-sm text-muted-foreground">
              {!isLoading && totalItems > 0 && (
                <span>
                  Page {currentPage} of {totalPages} • {totalItems} total items
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="container mx-auto px-4 py-8">
        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : allEmbeddings.length > 0 ? (
          <>
            {/* Items grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 mb-8">
              {allEmbeddings.map((item) => (
                <div
                  key={item.id}
                  className="group relative bg-card rounded-lg overflow-hidden border border-border hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/10"
                >
                  {/* Media preview */}
                  <div className="aspect-square relative overflow-hidden bg-secondary/10">
                    {getMediaPreview(item)}
                    
                    {/* Gradient overlay on hover */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                    
                    {/* Media type indicator */}
                    <div className="absolute top-2 left-2 p-1.5 bg-black/60 backdrop-blur-sm rounded-full text-white pointer-events-none">
                      {getMediaIcon(item)}
                    </div>
                  </div>

                  {/* Info section */}
                  <div className="p-3">
                    {/* Text preview if available */}
                    {item.text && (
                      <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
                        {item.text}
                      </p>
                    )}
                    
                    {/* ID and date */}
                    <div className="space-y-1">
                      <p className="text-xs text-muted-foreground font-mono truncate" title={item.id}>
                        ID: {item.id.length > 10 ? `${item.id.slice(0, 8)}...` : item.id}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(item.createdAt).toLocaleDateString()}
                      </p>
                    </div>

                    {/* Actions */}
                    <div className="mt-3 flex items-center justify-between">
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => handleDelete(item.id)}
                        className="flex-1"
                      >
                        <Trash2 className="w-3.5 h-3.5 mr-1" />
                        Delete
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination controls */}
            <div className="flex items-center justify-center gap-2">
              <Button
                variant="outline"
                size="icon"
                onClick={() => goToPage(1)}
                disabled={currentPage === 1}
                className="h-9 w-9"
              >
                <ChevronsLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage === 1}
                className="h-9 w-9"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>

              {/* Page numbers */}
              <div className="flex items-center gap-1">
                {currentPage > 4 && totalPages > 7 && (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => goToPage(1)}
                      className="h-9 min-w-[36px] px-2"
                    >
                      1
                    </Button>
                    <span className="px-1">...</span>
                  </>
                )}

                {getPageNumbers().map((page) => (
                  <Button
                    key={page}
                    variant={currentPage === page ? "default" : "outline"}
                    size="sm"
                    onClick={() => goToPage(page)}
                    className={cn(
                      "h-9 min-w-[36px] px-2",
                      currentPage === page && "bg-gradient-to-r from-purple-500 to-pink-500 border-0"
                    )}
                  >
                    {page}
                  </Button>
                ))}

                {currentPage < totalPages - 3 && totalPages > 7 && (
                  <>
                    <span className="px-1">...</span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => goToPage(totalPages)}
                      className="h-9 min-w-[36px] px-2"
                    >
                      {totalPages}
                    </Button>
                  </>
                )}
              </div>

              <Button
                variant="outline"
                size="icon"
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage === totalPages}
                className="h-9 w-9"
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={() => goToPage(totalPages)}
                disabled={currentPage === totalPages}
                className="h-9 w-9"
              >
                <ChevronsRight className="h-4 w-4" />
              </Button>
            </div>
          </>
        ) : (
          /* Empty state */
          <div className="text-center py-20">
            <Video className="w-12 h-12 mx-auto mb-4 text-muted-foreground/50" />
            <p className="text-lg text-muted-foreground">No embeddings found</p>
            <p className="text-sm text-muted-foreground mt-2">
              Upload some videos or images to get started
            </p>
          </div>
        )}

        {/* Info about pagination */}
        {allEmbeddings.length >= 100 && (
          <div className="text-center py-8">
            <p className="text-sm text-muted-foreground">
              Showing first 100 items. To load more, API pagination support is needed.
            </p>
          </div>
        )}
      </div>

      {/* Delete confirmation dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Embedding</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete this embedding? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
              disabled={!!deletingId}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={confirmDelete}
              disabled={!!deletingId}
            >
              {deletingId ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                'Delete'
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

