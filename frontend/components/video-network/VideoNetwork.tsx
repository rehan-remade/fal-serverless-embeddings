'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { Stage, Layer, Circle, Line, Text, Group, Rect } from 'react-konva';
import Konva from 'konva';
import { trpc } from '@/lib/trpc/client';
import { Card } from '@/components/ui/card';
import { Loader2, X } from 'lucide-react';

interface Video {
  id: string;
  videoUrl: string | null;
  text: string | null;
  position: [number, number];
  distance?: number;
}

interface VideoNode extends Video {
  x: number;
  y: number;
  radius: number;
  color: string;
}

export function VideoNetwork() {
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [selectedVideo, setSelectedVideo] = useState<Video | null>(null);
  const [hoveredVideo, setHoveredVideo] = useState<Video | null>(null);
  const [stagePosition, setStagePosition] = useState({ x: 0, y: 0 });
  const [stageScale, setStageScale] = useState(1);
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch all videos with positions
  const { data: videoData, isLoading } = trpc.embedding.getAllWithPositions.useQuery({
    limit: 300,
    dimensions: '2d'
  });

  // Fetch similar videos when one is selected
  const { data: similarData } = trpc.embedding.findSimilar.useQuery(
    {
      videoId: selectedVideo?.id || '',
      limit: 20,
      distanceThreshold: 10
    },
    {
      enabled: !!selectedVideo?.id
    }
  );

  // Update dimensions on mount and resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Process video data for visualization
  const processedVideos: VideoNode[] = videoData?.videos.map(video => {
    const bounds = videoData.bounds;
    
    // Normalize positions to fit the canvas
    const x = ((video.position[0] - bounds.min[0]) / (bounds.max[0] - bounds.min[0])) * (dimensions.width - 100) + 50;
    const y = ((video.position[1] - bounds.min[1]) / (bounds.max[1] - bounds.min[1])) * (dimensions.height - 100) + 50;
    
    // Determine node properties based on selection state
    let radius = 6;
    let color = '#6b7280';
    
    if (selectedVideo?.id === video.id) {
      radius = 15;
      color = '#ef4444';
    } else if (similarData?.similarVideos.some(v => v.id === video.id)) {
      radius = 10;
      color = '#3b82f6';
    }
    
    return {
      ...video,
      x,
      y,
      radius,
      color
    };
  }) || [];

  // Handle zoom
  const handleWheel = (e: Konva.KonvaEventObject<WheelEvent>) => {
    e.evt.preventDefault();
    
    const scaleBy = 1.1;
    const stage = e.target.getStage();
    if (!stage) return;
    
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    
    const direction = e.evt.deltaY > 0 ? -1 : 1;
    const newScale = direction > 0 ? oldScale * scaleBy : oldScale / scaleBy;
    
    // Limit zoom
    if (newScale < 0.5 || newScale > 5) return;
    
    setStageScale(newScale);
    setStagePosition({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  // Handle drag
  const handleDragEnd = (e: Konva.KonvaEventObject<DragEvent>) => {
    setStagePosition({
      x: e.target.x(),
      y: e.target.y(),
    });
  };

  // Get connection lines for similar videos
  const getConnectionLines = () => {
    if (!selectedVideo || !similarData) return [];
    
    const sourceNode = processedVideos.find(v => v.id === selectedVideo.id);
    if (!sourceNode) return [];
    
    return similarData.similarVideos.map(similarVideo => {
      const targetNode = processedVideos.find(v => v.id === similarVideo.id);
      if (!targetNode) return null;
      
      const maxDistance = Math.max(...similarData.similarVideos.map(v => v.distance));
      const normalizedDistance = similarVideo.distance / maxDistance;
      
      return {
        points: [sourceNode.x, sourceNode.y, targetNode.x, targetNode.y],
        stroke: '#3b82f6',
        strokeWidth: Math.max(1, 5 * (1 - normalizedDistance)),
        opacity: 0.2 + 0.6 * (1 - normalizedDistance),
        dash: similarVideo.distance > 5 ? [5, 5] : undefined,
        distance: similarVideo.distance,
        midX: (sourceNode.x + targetNode.x) / 2,
        midY: (sourceNode.y + targetNode.y) / 2,
      };
    }).filter(Boolean);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  const connectionLines = getConnectionLines();

  return (
    <div className="relative w-full h-screen bg-gray-900" ref={containerRef}>
      <Stage
        width={dimensions.width}
        height={dimensions.height}
        draggable
        onWheel={handleWheel}
        onDragEnd={handleDragEnd}
        x={stagePosition.x}
        y={stagePosition.y}
        scaleX={stageScale}
        scaleY={stageScale}
      >
        <Layer>
          {/* Connection lines */}
          {connectionLines.map((line, i) => (
            <Group key={`connection-${i}`}>
              <Line
                points={line?.points}
                stroke={line?.stroke}
                strokeWidth={line?.strokeWidth}
                opacity={line?.opacity}
                dash={line?.dash}
              />
              {/* Distance label */}
              <Text
                x={line?.midX - 30}
                y={line?.midY - 8}
                text={`L2: ${line?.distance.toFixed(2)}`}
                fontSize={10}
                fill="#64748b"
                width={60}
                align="center"
              />
            </Group>
          ))}

          {/* Video nodes */}
          {processedVideos.map((video) => (
            <Group key={video.id}>
              <Circle
                x={video.x}
                y={video.y}
                radius={video.radius}
                fill={video.color}
                stroke="#fff"
                strokeWidth={2}
                shadowBlur={selectedVideo?.id === video.id ? 10 : 0}
                shadowColor={video.color}
                onMouseEnter={(e) => {
                  const container = e.target.getStage()?.container();
                  if (container) {
                    container.style.cursor = 'pointer';
                  }
                  setHoveredVideo(video);
                }}
                onMouseLeave={(e) => {
                  const container = e.target.getStage()?.container();
                  if (container) {
                    container.style.cursor = 'default';
                  }
                  setHoveredVideo(null);
                }}
                onClick={() => setSelectedVideo(video)}
              />

              {/* Hover tooltip */}
              {hoveredVideo?.id === video.id && (
                <Group>
                  <Rect
                    x={video.x - 100}
                    y={video.y - 40}
                    width={200}
                    height={25}
                    fill="rgba(0, 0, 0, 0.8)"
                    cornerRadius={3}
                  />
                  <Text
                    x={video.x - 95}
                    y={video.y - 35}
                    text={video.text ? 
                      (video.text.length > 30 ? video.text.substring(0, 30) + '...' : video.text) : 
                      'No description'
                    }
                    fontSize={12}
                    fill="white"
                    width={190}
                    align="center"
                  />
                </Group>
              )}
            </Group>
          ))}
        </Layer>
      </Stage>

      {/* Control Panel */}
      <Card className="absolute top-4 left-4 p-4 bg-black/80 backdrop-blur">
        <h2 className="text-lg font-semibold mb-2">Video Network</h2>
        <p className="text-sm text-gray-400 mb-4">
          Click a video to see similar ones. L2 distance shown on connections.
        </p>
        <div className="text-xs text-gray-500 space-y-1">
          <p>• Scroll to zoom</p>
          <p>• Drag to pan</p>
          <p>• Click videos to explore</p>
        </div>
        {selectedVideo && (
          <div className="mt-4 pt-4 border-t border-gray-700 space-y-2">
            <p className="text-sm font-medium">Selected Video</p>
            <p className="text-xs text-gray-400">{selectedVideo.text || 'No description'}</p>
            {similarData && (
              <p className="text-xs text-blue-400">
                {similarData.similarVideos.length} similar videos found
              </p>
            )}
          </div>
        )}
      </Card>

      {/* Legend */}
      <Card className="absolute bottom-4 left-4 p-4 bg-black/80 backdrop-blur">
        <h3 className="text-sm font-semibold mb-2">Legend</h3>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span className="text-xs">Selected Video</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-blue-500"></div>
            <span className="text-xs">Similar Videos</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-gray-500"></div>
            <span className="text-xs">Other Videos</span>
          </div>
        </div>
      </Card>

      {/* Video Preview Modal */}
      {selectedVideo?.videoUrl && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
          onClick={() => setSelectedVideo(null)}
        >
          <div
            className="bg-gray-900 p-6 rounded-lg max-w-4xl max-h-[90vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-start mb-4">
              <h3 className="text-xl font-bold">{selectedVideo.text || 'Video'}</h3>
              <button
                onClick={() => setSelectedVideo(null)}
                className="p-1 hover:bg-gray-800 rounded"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <video
              src={selectedVideo.videoUrl}
              controls
              className="w-full rounded"
              style={{ maxHeight: '60vh' }}
            />
            {similarData && (
              <div className="mt-4">
                <h4 className="font-semibold mb-2">Similar Videos (L2 Distance)</h4>
                <div className="grid grid-cols-4 gap-2">
                  {similarData.similarVideos.slice(0, 8).map((video) => (
                    <div key={video.id} className="relative group">
                      <div className="aspect-video bg-gray-800 rounded overflow-hidden">
                        {video.videoUrl ? (
                          <video
                            src={video.videoUrl}
                            className="w-full h-full object-cover"
                            muted
                            onMouseEnter={(e) => e.currentTarget.play()}
                            onMouseLeave={(e) => {
                              e.currentTarget.pause();
                              e.currentTarget.currentTime = 0;
                            }}
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center text-gray-600">
                            No video
                          </div>
                        )}
                      </div>
                      <p className="text-xs text-gray-400 mt-1">L2: {video.distance.toFixed(2)}</p>
                      <p className="text-xs text-gray-500 truncate">{video.text || 'No description'}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
