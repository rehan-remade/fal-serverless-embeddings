import { EmbeddingForm } from '@/components/embeddings/embedding-form';

export default function EmbeddingPage() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-4xl font-bold text-center mb-2">Create Embeddings</h1>
          <p className="text-center text-muted-foreground mb-8">
            Generate embeddings from text, images, or videos using VLM2Vec
          </p>
          
          <EmbeddingForm />
        </div>
      </div>
    </div>
  );
}
