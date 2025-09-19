import { EmbeddingForm } from '../components/embeddings/embedding-form';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      <div className="container mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold text-center mb-2">Shared Embedding Space</h1>
        <p className="text-center text-gray-400 mb-12">
          Create multimodal embeddings from text, images, and videos
        </p>

        <div className="max-w-3xl mx-auto">
          <EmbeddingForm />
          
          <div className="mt-12">
            <h2 className="text-2xl font-semibold mb-6">Recent Embeddings</h2>
          </div>
        </div>
      </div>
    </div>
  );
}
