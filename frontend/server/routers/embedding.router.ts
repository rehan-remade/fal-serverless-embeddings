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
});
