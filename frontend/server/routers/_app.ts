import { router } from '../trpc';
import { embeddingRouter } from './embedding.router';

export const appRouter = router({
  embedding: embeddingRouter,
});

export type AppRouter = typeof appRouter;
