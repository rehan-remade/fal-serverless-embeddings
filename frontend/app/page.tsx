import { EmbeddingForm } from '../components/embeddings/embedding-form';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      <div className="container mx-auto px-4 py-12">
        <h1 className="text-4xl font-bold text-center mb-2">Shared Embedding Space</h1>
        <p className="text-center text-gray-400 mb-4">
          Create multimodal embeddings from text, images, and videos
        </p>
        
        {/* Add navigation button */}
        <div className="text-center mb-12">
          <Link
            href="/network"
            className="inline-flex items-center px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Explore Video Network
          </Link>
        </div>

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
