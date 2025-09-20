import { z } from 'zod';
import { router, publicProcedure } from '@/server/trpc';
import { 
  createEmbeddingSchema, 
  searchEmbeddingsSchema,
  embeddingSchema,
  CreateEmbeddingInput,
  SearchEmbeddingsInput,
  ListEmbeddingsInput
} from '@/lib/schemas/embedding.schema';
import { embeddingService } from '../../lib/services/embedding.service';
import { TRPCError } from '@trpc/server';

export const embeddingRouter = router({
  create: publicProcedure
    .input(createEmbeddingSchema)
    .output(z.object({
      success: z.boolean(),
      id: z.string(),
      dimension: z.number()
    }))
    .mutation(async ({ input }: { input: CreateEmbeddingInput }) => {
      try {
        const result = await embeddingService.createEmbedding(input);
        return {
          success: true,
          ...result
        };
      } catch (error) {
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Failed to create embedding',
        });
      }
    }),

  search: publicProcedure
    .input(searchEmbeddingsSchema)
    .output(z.object({
      results: z.array(embeddingSchema.extend({ distance: z.number() }))
    }))
    .mutation(async ({ input }: { input: SearchEmbeddingsInput }) => {
      try {
        // Generate embedding for search query
        const queryEmbedding = await embeddingService.generateQueryEmbedding({
          text: input.text,
          imageUrl: input.imageUrl,
          videoUrl: input.videoUrl,
        });

        // Search
        const results = await embeddingService.searchEmbeddings(queryEmbedding, input.limit);

        return { results };
      } catch (error) {
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Search failed',
        });
      }
    }),

  list: publicProcedure
    .input(z.object({
      limit: z.number().min(1).max(100).default(50)
    }))
    .output(z.object({
      embeddings: z.array(embeddingSchema)
    }))
    .query(async ({ input }: { input: ListEmbeddingsInput }) => {
      try {
        const embeddings = await embeddingService.getRecentEmbeddings(input.limit);
        return { embeddings };
      } catch (error) {
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Failed to fetch embeddings',
        });
      }
    }),

  findSimilar: publicProcedure
    .input(z.object({
      videoId: z.string(),
      limit: z.number().min(5).max(50).default(20),
      distanceThreshold: z.number().optional() // Max L2 distance
    }))
    .output(z.object({
      sourceVideo: z.object({
        id: z.string(),
        text: z.string().nullable(),
        imageUrl: z.string().nullable(),
        videoUrl: z.string().nullable(),
        createdAt: z.date()
      }),
      similarVideos: z.array(z.object({
        id: z.string(),
        text: z.string().nullable(),
        imageUrl: z.string().nullable(),
        videoUrl: z.string().nullable(),
        createdAt: z.date(),
        distance: z.number()
      }))
    }))
    .query(async ({ input }) => {
      try {
        console.log('Finding similar videos for:', input.videoId);
        
        // Get the source video
        const sourceVideo = await embeddingService.getEmbeddingById(input.videoId);
        if (!sourceVideo) {
          throw new TRPCError({
            code: 'NOT_FOUND',
            message: `Video with ID ${input.videoId} not found`
          });
        }

        console.log('Source video found, searching for similar...');

        // Find similar videos using L2 distance
        const similarVideos = await embeddingService.searchEmbeddings(
          sourceVideo.embedding,
          input.limit + 1, // +1 to exclude self
          'l2',
          input.distanceThreshold
        );

        console.log(`Found ${similarVideos.length} similar videos`);

        // Filter out the source video itself
        const filtered = similarVideos.filter(v => v.id !== input.videoId);

        // Return without embedding arrays to avoid serialization issues
        return {
          sourceVideo: {
            id: sourceVideo.id,
            text: sourceVideo.text,
            imageUrl: sourceVideo.imageUrl,
            videoUrl: sourceVideo.videoUrl,
            createdAt: sourceVideo.createdAt
          },
          similarVideos: filtered.slice(0, input.limit).map(v => ({
            id: v.id,
            text: v.text,
            imageUrl: v.imageUrl,
            videoUrl: v.videoUrl,
            createdAt: v.createdAt,
            distance: v.distance
          }))
        };
      } catch (error) {
        console.error('Error in findSimilar:', error);
        if (error instanceof TRPCError) {
          throw error;
        }
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Failed to find similar videos',
        });
      }
    }),

  getAllWithPositions: publicProcedure
    .input(z.object({
      limit: z.number().min(50).max(500).default(300),
      dimensions: z.enum(['2d', '3d']).default('2d')
    }))
    .output(z.object({
      videos: z.array(z.object({
        id: z.string(),
        videoUrl: z.string().nullable(),
        text: z.string().nullable(),
        imageUrl: z.string().nullable(),
        position: z.array(z.number()),
        cluster: z.number().optional()
      })),
      bounds: z.object({
        min: z.array(z.number()),
        max: z.array(z.number())
      })
    }))
    .query(async ({ input }) => {
      try {
        const embeddings = await embeddingService.getRecentEmbeddings(input.limit);
        
        // Reduce dimensions using UMAP (you'll need to implement this)
        const positions = await embeddingService.reduceEmbeddingDimensions(
          embeddings,
          input.dimensions === '2d' ? 2 : 3
        );

        return {
          videos: embeddings.map((e, i) => ({
            id: e.id,
            videoUrl: e.videoUrl,
            text: e.text,
            imageUrl: e.imageUrl,
            position: positions[i],
            cluster: 0 // You can add clustering later
          })),
          bounds: {
            min: positions.reduce((min, p) => p.map((v, i) => Math.min(v, min[i] || v))),
            max: positions.reduce((max, p) => p.map((v, i) => Math.max(v, max[i] || v)))
          }
        };
      } catch (error) {
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Failed to get video positions',
        });
      }
    })
});
