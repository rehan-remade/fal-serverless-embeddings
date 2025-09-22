import { fal } from '@fal-ai/client';
import * as lancedb from "@lancedb/lancedb"
import { v4 as uuidv4 } from 'uuid';
import type { Embedding, CreateEmbeddingInput, SearchResult } from '../schemas/embedding.schema';
import { Schema, Field, Float32, FixedSizeList, Utf8, Float64 } from "apache-arrow";

// Configure fal client - set this via environment variable instead
// fal automatically uses FAL_KEY environment variable

export class EmbeddingService {
  private db: any = null;  // LanceDB connection type
  private table: any = null;  // LanceDB table type

  constructor() {}

  private async getDb(): Promise<any> {
    if (!this.db) {
      this.db = await lancedb.connect({
        uri: process.env.LANCEDB_URI || "db://serverless-example-custom-embeddings-rk1m94",
        apiKey: process.env.LANCEDB_API_KEY!,
        region: "us-east-1"
      });
    }
    return this.db;
  }

  private async getTable(): Promise<any> {
    if (!this.table) {
      const db = await this.getDb();
      const tableNames = await db.tableNames();
      
      if (tableNames.includes('embeddings')) {
        this.table = await db.openTable('embeddings');
      } else {
        // Create table with schema
        const schema = new Schema([
          new Field("id", new Utf8()),
          new Field(
            "embedding",
            new FixedSizeList(1536, new Field("float32", new Float32())),  // VLM2Vec-Qwen2VL-2B outputs 1536
          ),
          new Field("text", new Utf8()),
          new Field("imageUrl", new Utf8()),
          new Field("videoUrl", new Utf8()),
          new Field("createdAt", new Float64()), // Store as timestamp
        ]);

        this.table = await db.createTable({
          name: 'embeddings',
          schema,
          data: []
        });
      }
    }
    return this.table;
  }

  async createEmbedding(input: CreateEmbeddingInput): Promise<{ id: string; dimension: number }> {
    // Call VLM2Vec embedding endpoint
    const { data } = await fal.subscribe(process.env.FAL_MODEL_ID!, {
      input: {
        text: input.text,
        image_url: input.imageUrl,
        video_url: input.videoUrl,
      }
    }) as { data: { embedding: number[]; dimension: number } };

    // Store in LanceDB
    const table = await this.getTable();
    const record = {
      id: uuidv4(),
      embedding: data.embedding,
      text: input.text || "",
      imageUrl: input.imageUrl || "",
      videoUrl: input.videoUrl || "",
      createdAt: Date.now(), // Store as timestamp
    };

    await table.add([record]);

    return {
      id: record.id,
      dimension: data.dimension
    };
  }

  async searchEmbeddings(
    queryEmbedding: number[], 
    limit: number,
    distanceType: 'l2' | 'cosine' | 'dot' = 'l2',
    distanceThreshold?: number
  ): Promise<SearchResult[]> {
    const table = await this.getTable();
    
    try {
      // Basic search without threshold - let client filter if needed
      const results = await table
        .search(queryEmbedding)
        .distanceType(distanceType)
        .limit(limit)
        .toArray();

      console.log(`Search found ${results.length} results`);

      return results.map((r: any) => ({
        id: r.id,
        embedding: r.embedding,
        text: r.text || null,
        imageUrl: r.imageUrl || null,
        videoUrl: r.videoUrl || null,
        createdAt: new Date(r.createdAt),
        distance: r._distance
      }));
    } catch (error) {
      console.error('Error searching embeddings:', error);
      return [];
    }
  }

  async getRecentEmbeddings(limit: number = 50): Promise<Embedding[]> {
    const table = await this.getTable();
    
    // Query all records and sort by createdAt
    const allRecords = await table.query().toArray();
    
    return allRecords
      .map((r: any) => ({
        id: r.id,
        embedding: r.embedding,
        text: r.text || null,
        imageUrl: r.imageUrl || null,
        videoUrl: r.videoUrl || null,
        createdAt: new Date(r.createdAt)
      }))
      .sort((a: any, b: any) => b.createdAt.getTime() - a.createdAt.getTime())
      .slice(0, limit);
  }

  async generateQueryEmbedding(input: Omit<CreateEmbeddingInput, 'metadata'>): Promise<number[]> {
    const { data } = await fal.subscribe(process.env.FAL_MODEL_ID!, {
      input: {
        text: input.text,
        image_url: input.imageUrl,
        video_url: input.videoUrl,
      }
    }) as { data: { embedding: number[]; dimension: number } };

    return data.embedding;
  }

  async getEmbeddingById(id: string): Promise<Embedding | null> {
    const table = await this.getTable();
    
    try {
      // Use parameterized query to avoid SQL injection and syntax issues
      const results = await table
        .query()
        .where(`id = "${id}"`) // Use double quotes for string literals in LanceDB
        .toArray();
      
      if (results.length === 0) return null;
      
      const r = results[0];
      return {
        id: r.id,
        embedding: r.embedding,
        text: r.text || null,
        imageUrl: r.imageUrl || null,
        videoUrl: r.videoUrl || null,
        createdAt: new Date(r.createdAt)
      };
    } catch (error) {
      console.error('Error fetching embedding by ID:', error);
      return null;
    }
  }

  async getRandomEmbeddings(limit: number = 50): Promise<Embedding[]> {
    const table = await this.getTable();
    
    try {
      // Query all records
      const allRecords = await table.query().toArray();
      
      // Shuffle array using Fisher-Yates algorithm
      const shuffled = [...allRecords];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      
      // Return limited random results
      return shuffled
        .slice(0, limit)
        .map((r: any) => ({
          id: r.id,
          embedding: r.embedding,
          text: r.text || null,
          imageUrl: r.imageUrl || null,
          videoUrl: r.videoUrl || null,
          createdAt: new Date(r.createdAt)
        }));
    } catch (error) {
      console.error('Error fetching random embeddings:', error);
      return [];
    }
  }

  // Simple dimension reduction using PCA (for demo)
  // In production, you'd want to use UMAP or t-SNE
  async reduceEmbeddingDimensions(
    embeddings: Embedding[],
    targetDimensions: number
  ): Promise<number[][]> {
    // For now, return random positions
    // In production, implement proper UMAP/t-SNE/PCA
    return embeddings.map(() => {
      const positions = [];
      for (let i = 0; i < targetDimensions; i++) {
        positions.push((Math.random() - 0.5) * 20);
      }
      return positions;
    });
  }
}

// Singleton instance
export const embeddingService = new EmbeddingService();
