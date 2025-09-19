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
            new FixedSizeList(5120, new Field("float32", new Float32())),
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

  async searchEmbeddings(queryEmbedding: number[], limit: number): Promise<SearchResult[]> {
    const table = await this.getTable();
    
    const results = await table
      .search(queryEmbedding)
      .limit(limit)
      .toArray();

    return results.map((r: any) => ({
      id: r.id,
      embedding: r.embedding,
      text: r.text || null,
      imageUrl: r.imageUrl || null,
      videoUrl: r.videoUrl || null,
      createdAt: new Date(r.createdAt),
      distance: r._distance
    }));
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
}

// Singleton instance
export const embeddingService = new EmbeddingService();
