using Microsoft.Extensions.AI;

namespace LlamaVisionPOC
{
    public class SimpleChat
    {
        public static async Task RunAsync()
        {

            IChatClient client =
                new OllamaChatClient(new Uri("http://localhost:11434/"), "phi3");

            List<ChatMessage> messages =
            [
                new ChatMessage(ChatRole.System, "You are a frendly asistant. Answer in shorts sentences."),
                new ChatMessage(ChatRole.User, "Hello! What is AI?"),
            ];

            //IEmbeddingGenerator<String, Embedding<Single>> generator =
            //    new OllamaEmbeddingGenerator(new Uri("http://localhost:11434/"), "nomic-embed-text");

            //var aa = await generator.GenerateAsync(messages.Select(m => m.Text).ToArray());

            var response = await client.CompleteAsync<AnswerObject>(messages);

            Console.WriteLine($"Message: {response.Result.Message}");
            Console.WriteLine($"Intent: {response.Result.Intent}");
            Console.WriteLine($"Confidence: {response.Result.Confidence}");
            Console.WriteLine($"tokens: {response.Usage!.TotalTokenCount ?? -1}");
        }

        public class AnswerObject
        {
            public string? Message { get; set; }
            public string? Intent { get; set; }
            public string? Confidence { get; set; }
        }
    }
}
