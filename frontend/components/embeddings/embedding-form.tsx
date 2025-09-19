'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Loader2 } from 'lucide-react';
import { trpc } from '@/lib/trpc/client';
import { createEmbeddingSchema, type CreateEmbeddingInput } from '@/lib/schemas/embedding.schema';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { toast } from 'sonner';

export function EmbeddingForm() {
  const utils = trpc.useUtils();

  const form = useForm<CreateEmbeddingInput>({
    resolver: zodResolver(createEmbeddingSchema),
    defaultValues: {
      text: '',
      imageUrl: '',
      videoUrl: '',
    },
    mode: 'onTouched',
  });

  const createEmbedding = trpc.embedding.create.useMutation({
    onSuccess: () => {
      form.reset();
      utils.embedding.list.invalidate();
      toast.success("Embedding created successfully!");
    },
    onError: (error: any) => {
      toast.error(error.message);
    },
  });

  const onSubmit = async (data: CreateEmbeddingInput) => {
    // Remove empty string values so backend gets undefined for unused fields
    const cleanedData: CreateEmbeddingInput = {
      text: data.text?.trim() ? data.text : undefined,
      imageUrl: data.imageUrl?.trim() ? data.imageUrl : undefined,
      videoUrl: data.videoUrl?.trim() ? data.videoUrl : undefined,
    };
    
    // Additional validation: ensure at least one field has a value
    if (!cleanedData.text && !cleanedData.imageUrl && !cleanedData.videoUrl) {
      toast.error("At least one input (text, imageUrl, or videoUrl) is required");
      return;
    }
    
    createEmbedding.mutate(cleanedData);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create Embedding</CardTitle>
        <CardDescription>
          Generate embeddings from text, images, or videos. You may provide any combination, but at least one is required.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <FormField
              control={form.control}
              name="text"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Text Input</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Enter text to embed..."
                      className="min-h-[100px]"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>
                    Enter any text you want to create an embedding for (optional)
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="imageUrl"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Image URL</FormLabel>
                    <FormControl>
                      <Input placeholder="https://your-storage.com/image.png" {...field} />
                    </FormControl>
                    <FormDescription>
                      Enter a direct URL to an image file (optional)
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="videoUrl"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Video URL</FormLabel>
                    <FormControl>
                      <Input placeholder="https://your-storage.com/video.mp4" {...field} />
                    </FormControl>
                    <FormDescription>
                      Enter a direct URL to a video file (optional)
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="text-sm text-gray-400">
              <span>
                <b>Note:</b> You may fill in any combination of the above fields, but at least one is required.
              </span>
            </div>

            <Button
              type="submit"
              className="w-full"
              disabled={createEmbedding.isPending}
            >
              {createEmbedding.isPending && (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              )}
              Create Embedding
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  );
}
