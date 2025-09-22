import { Skeleton } from '@/components/ui/skeleton';

interface LoadingGridProps {
  count?: number;
}

// Predefined heights to ensure consistency between server and client
const PREDEFINED_HEIGHTS = [
  320, 280, 390, 350, 410, 290, 370, 330, 400, 310, 360, 420,
  300, 380, 340, 430, 270, 390, 350, 320, 410, 290, 370, 400
];

export const LoadingGrid: React.FC<LoadingGridProps> = ({ count = 12 }) => {
  return (
    <div className="container mx-auto">
      <div className="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4">
        {Array.from({ length: count }).map((_, index) => {
          // Use modulo to cycle through predefined heights
          const height = PREDEFINED_HEIGHTS[index % PREDEFINED_HEIGHTS.length];
          
          return (
            <div
              key={index}
              className="break-inside-avoid mb-4"
              style={{
                animationDelay: `${index * 50}ms`,
                animation: 'fadeInUp 0.5s ease-out forwards',
                opacity: 0
              }}
            >
              <div className="bg-card border border-border/50 rounded-xl overflow-hidden">
                <Skeleton 
                  className="w-full"
                  style={{ height: `${height}px` }}
                />
                <div className="p-4 space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
