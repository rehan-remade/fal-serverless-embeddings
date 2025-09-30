import { z } from 'zod';
import { router, publicProcedure } from '@/server/trpc';
import { 
  createEmbeddingSchema, 
  searchEmbeddingsSchema,
  CreateEmbeddingInput,
  SearchEmbeddingsInput
} from '@/lib/schemas/embedding.schema';
import { embeddingService } from '../../lib/services/embedding.service';
import { TRPCError } from '@trpc/server';
import { fal } from '@fal-ai/client';

export const embeddingRouter = router({
  uploadVideo: publicProcedure
    .input(z.object({
      fileBase64: z.string(),
      fileName: z.string(),
      mimeType: z.string()
    }))
    .output(z.object({
      url: z.string()
    }))
    .mutation(async ({ input }) => {
      try {
        // Convert base64 to buffer
        const buffer = Buffer.from(input.fileBase64, 'base64');
        
        // Create a Blob from the buffer
        const blob = new Blob([buffer], { type: input.mimeType });
        
        // Create a File object
        const file = new File([blob], input.fileName, { type: input.mimeType });
        
        // Upload to fal.storage
        const url = await fal.storage.upload(file);
        
        return { url };
      } catch (error) {
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Failed to upload video',
        });
      }
    }),

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
      results: z.array(z.object({
        id: z.string(),
        text: z.string().nullable(),
        imageUrl: z.string().nullable(), 
        videoUrl: z.string().nullable(),
        createdAt: z.date(),
        distance: z.number()
      }))
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

        // Return results without the embedding array
        return { 
          results: results.map(r => ({
            id: r.id,
            text: r.text,
            imageUrl: r.imageUrl,
            videoUrl: r.videoUrl,
            createdAt: r.createdAt,
            distance: r.distance
          }))
        };
      } catch (error) {
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Search failed',
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

  getRandom: publicProcedure
    .input(z.object({
      limit: z.number().min(1).max(100).default(50)
    }))
    .output(z.object({
      embeddings: z.array(z.object({
        id: z.string(),
        text: z.string().nullable(),
        imageUrl: z.string().nullable(),
        videoUrl: z.string().nullable(),
        createdAt: z.date()
      }))
    }))
    .query(async ({ input }) => {
      try {
        const embeddings = await embeddingService.getRandomEmbeddings(input.limit);
        
        // Return without embedding arrays to avoid serialization issues
        return {
          embeddings: embeddings.map(e => ({
            id: e.id,
            text: e.text,
            imageUrl: e.imageUrl,
            videoUrl: e.videoUrl,
            createdAt: e.createdAt
          }))
        };
      } catch (error) {
        throw new TRPCError({
          code: 'INTERNAL_SERVER_ERROR',
          message: error instanceof Error ? error.message : 'Failed to fetch random embeddings',
        });
      }
    }),
});
