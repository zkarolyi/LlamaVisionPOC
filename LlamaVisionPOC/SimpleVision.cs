using Microsoft.Extensions.AI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LlamaVisionPOC
{
    public class SimpleVision
    {
        public static async Task RunAsync()
        {
            IChatClient client =
                new OllamaChatClient(new Uri("http://localhost:11434/"), "llama3.2-vision");

            string[] imageFiles = Directory.GetFiles("../../../images", "*.jpg");

            foreach (var file in imageFiles)
            {
                var msg = new ChatMessage(ChatRole.User, "Is is a measurement tool? Answer with 'yes','probably','no' only.");
                msg.Contents.Add(new ImageContent(File.ReadAllBytes(file), "image/jpg"));

                var response = await client.CompleteAsync(new[] { msg });

                Console.WriteLine($"{DateTime.Now.ToString("hh:mm:ss")} File: {file}: Message: {response.Message} ({response.Usage!.TotalTokenCount})");
            }
        }
    }
}
