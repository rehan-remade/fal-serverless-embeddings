import { z } from 'zod';

// Input schemas
export const createEmbeddingSchema = z.object({
  text: z.string().optional(),
  imageUrl: z.url().optional().or(z.literal('')),
  videoUrl: z.url().optional().or(z.literal('')),
}).refine(
  (data) => {
    const hasText = data.text && data.text.trim().length > 0;
    const hasImageUrl = data.imageUrl && data.imageUrl.trim().length > 0;
    const hasVideoUrl = data.videoUrl && data.videoUrl.trim().length > 0;
    return hasText || hasImageUrl || hasVideoUrl;
  },
  { message: "At least one input (text, imageUrl, or videoUrl) is required" }
);

export const searchEmbeddingsSchema = z.object({
  text: z.string().optional(),
  imageUrl: z.url().optional(),
  videoUrl: z.url().optional(),
  limit: z.number().min(1).max(100).default(10),
}).refine(
  (data) => data.text || data.imageUrl || data.videoUrl,
  { message: "At least one search input is required" }
);

export const listEmbeddingsSchema = z.object({
  limit: z.number().min(1).max(100).default(50)
});

// Output schemas
export const embeddingSchema = z.object({
  id: z.uuid(),
  embedding: z.array(z.number()),
  text: z.string().nullable(),
  imageUrl: z.url().nullable(),
  videoUrl: z.url().nullable(),
  createdAt: z.date(),
});

export const embeddingResponseSchema = z.object({
  success: z.boolean(),
  id: z.uuid(),
  dimension: z.number(),
});

export const searchResultSchema = embeddingSchema.extend({
  distance: z.number(),
});

// Types
export type CreateEmbeddingInput = z.infer<typeof createEmbeddingSchema>;
export type SearchEmbeddingsInput = z.infer<typeof searchEmbeddingsSchema>;
export type Embedding = z.infer<typeof embeddingSchema>;
export type EmbeddingResponse = z.infer<typeof embeddingResponseSchema>;
export type SearchResult = z.infer<typeof searchResultSchema>;
export type ListEmbeddingsInput = z.infer<typeof listEmbeddingsSchema>;