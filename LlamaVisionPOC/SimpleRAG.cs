using Microsoft.Extensions.AI;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using System.Diagnostics;
using static LlamaVisionPOC.SimpleChat;

namespace LlamaVisionPOC
{
    public class SimpleRAG
    {
        public static async Task RunAsync()
        {

            IChatClient client =
                //new OllamaChatClient(new Uri("http://localhost:11434/"), "phi3");
                new OllamaChatClient(new Uri("http://localhost:11434/"), "deepseek-r1");

            IEmbeddingGenerator<string, Embedding<float>> generator =
                new OllamaEmbeddingGenerator(new Uri("http://localhost:11434/"), "nomic-embed-text"); //"all-minilm"

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Generating sample data...");

            var sampleText = new[] {
                "The capital city of France, Paris, is known for its iconic landmarks such as the Eiffel Tower and the Louvre Museum.",
                "Machine learning algorithms are often categorized into supervised, unsupervised, and reinforcement learning techniques.",
                "In 1969, the Apollo 11 mission successfully landed the first humans on the moon, marking a historic achievement for spacexploration.",
                "Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into oxygen and glucose.",
                "The Great Wall of China stretches over 13,000 miles and was built to protect against invasions during various dynasties.",
                "Albert Einstein developed the theory of relativity, which revolutionized modern physics.",
                "The Amazon Rainforest is home to millions of species, making it one of the most biodiverse places on Earth.",
                "Python is a versatile programming language widely used in data science and web development.",
                "The Pyramids of Giza in Egypt are among the Seven Wonders of the Ancient World.",
                "Electric cars are powered by batteries and produce no tailpipe emissions, making them environmentally friendly.",
                "Mount Everest, located in the Himalayas, is the highest mountain on Earth at 8,848 meters above sea level.",
                "The process of globalization has connected economies, cultures, and people across the world.",
                "Photosynthesis is essential for life on Earth, as it produces oxygen and serves as the base of the food chain.",
                "The Industrial Revolution began in the late 18th century and transformed manufacturing processes.",
                "Shakespeare's plays, such as 'Hamlet' and 'Macbeth,' are considered timeless works of literature.",
                "Renewable energy sources, such as wind and solar, are vital for reducing carbon emissions.",
                "The Great Barrier Reef is the largest coral reef system in the world, located off the coast of Australia.",
                "Cryptocurrencies like Bitcoin operate on blockchain technology, which ensures transparency and security.",
                "The human brain has approximately 86 billion neurons, making it one of the most complex organs.",
                "Julius Caesar was assassinated in 44 BCE, an event that led to the end of the Roman Republic.",
                "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
                "Marie Curie was the first woman to win a Nobel Prize and the only person to win it in two scientific fields.",
                "The Taj Mahal, located in India, is a UNESCO World Heritage Site and a symbol of love.",
                "Climate change is causing rising sea levels, extreme weather events, and biodiversity loss.",
                "Leonardo da Vinci's 'Mona Lisa' is one of the most famous paintings in the world.",
                "Vaccines work by stimulating the immune system to recognize and fight pathogens.",
                "The Hubble Space Telescope has provided stunning images and valuable data about the universe.",
                "The Great Wall of China was built over centuries to protect against invasions from the north.",
                "Artificial intelligence is being used in healthcare, finance, and autonomous vehicles.",
                "The Wright brothers achieved the first powered flight in 1903, marking the beginning of aviation.",
                "Water covers about 71% of Earth's surface, most of which is in the form of oceans.",
                "The periodic table organizes elements based on their atomic number and chemical properties.",
                "Social media platforms have transformed the way people communicate and share information.",
                "Martin Luther King Jr.'s 'I Have a Dream' speech is a cornerstone of the civil rights movement.",
                "Saturn is the sixth planet from the sun and is known for its prominent ring system.",
                "The Renaissance was a cultural movement that began in Italy during the 14th century.",
                "The first programmable computer, the Zuse Z3, was developed in 1941.",
                "Antarctica is the coldest continent on Earth, with temperatures dropping below -80°C.",
                "The Panama Canal connects the Atlantic and Pacific Oceans, significantly shortening travel time for ships.",
                "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones.",
                "Neil Armstrong was the first person to walk on the moon during the Apollo 11 mission in 1969.",
                "DNA, or deoxyribonucleic acid, carries genetic instructions for the development of living organisms.",
                "The Louvre Museum in Paris houses thousands of artworks, including the 'Mona Lisa' and the 'Venus de Milo'.",
                "Photosynthesis primarily occurs in the chloroplasts of plant cells.",
                "The internet has revolutionized how people access information, communicate, and shop.",
                "Mount Kilimanjaro in Tanzania is Africa's tallest mountain and a popular climbing destination.",
                "The process of mitosis allows cells to divide and reproduce for growth and repair.",
                "The Colosseum in Rome was used for gladiatorial games and public spectacles in ancient times.",
                "The Milky Way is the galaxy that contains our solar system.",
                "The four seasons result from Earth's axial tilt and its orbit around the sun.",
                "Ada Lovelace is often considered the first computer programmer due to her work on Charles Babbage's Analytical Engine.",
                "The term 'biodiversity' refers to the variety of life forms in a particular habitat or ecosystem.",
                "The Leaning Tower of Pisa is a famous architectural landmark in Italy.",
                "Einstein's equation, E=mc², explains the relationship between energy and mass.",
                "The Gutenberg printing press, invented in the 15th century, revolutionized the dissemination of knowledge." };

            List<RagData> sampleRagData = sampleText.Select((text, index) => new RagData { Key = index, Text = text }).ToList();

            var vectorStore = new InMemoryVectorStore();

            var samples = vectorStore.GetCollection<int, RagData>("sample");

            await samples.CreateCollectionIfNotExistsAsync();

            foreach (var item in sampleRagData)
            {
                item.Vector = await generator.GenerateEmbeddingVectorAsync(item.Text);
                await samples.UpsertAsync(item);
            }

            Console.WriteLine("Sample data generated.");
            Console.ResetColor();

            List<string> questions = new()
            {
                "Who was the first programmer?",
                "Is most of the earth's surface land?",
                "What is the capital city of France?",
                "Who developed the theory of relativity?",
                "What is the largest coral reef system in the world?",
                "What is the process by which green plants convert sunlight into oxygen?",
                "What is the first programmable computer called?",
                "What is the coldest continent on Earth?",
                "What is the term for the variety of life forms in a particular habitat?",
                "What is the galaxy that contains our solar system called?",
                "What is the series of numbers where each number is the sum of the two preceding ones?",
                "What is the relationship between energy and mass explained by Einstein's equation?"
            };

            foreach (var question in questions)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine(new string('-', 20));
                Console.WriteLine(question);
                Console.ResetColor();
                var embedding = await generator.GenerateEmbeddingVectorAsync(question);

                var searchOptions = new VectorSearchOptions()
                {
                    Top = 2,
                    VectorPropertyName = "Vector"
                };

                var results = await samples.VectorizedSearchAsync(embedding, searchOptions);

                var found = string.Empty;

                await foreach (var result in results.Results)
                {
                    Console.WriteLine($"Key: {result.Record.Key}");
                    Console.WriteLine($"Text:  {result.Record.Text}");
                    Console.WriteLine($"Score: {result.Score}");
                    Console.WriteLine();

                    if (result.Score > 0.5)
                    {
                        found += result.Record.Text + " ";
                    }
                }

                Console.ForegroundColor = ConsoleColor.Blue;
                Console.WriteLine($"Using: {found}");
                Console.ResetColor();

                List<ChatMessage> messages =
                [
                    new ChatMessage(ChatRole.System, "You are a frendly asistant. Explain your answer."),
                    new ChatMessage(ChatRole.Assistant, found.TrimEnd()),
                    new ChatMessage(ChatRole.User, question),
                ];

                ChatOptions options = new()
                {
                    ResponseFormat = ChatResponseFormat.Json,
                    Temperature = 0.7f
                };

                var response = await client.CompleteAsync<AnswerObject>(messages, options);

                Console.ForegroundColor = ConsoleColor.Red;
                AnswerObject? res;
                if (response.TryGetResult(out res))
                {
                    Console.WriteLine($"Message: {res.Message}");
                    Console.WriteLine($"Intent: {res.Intent}");
                    Console.WriteLine($"Confidence: {res.Confidence}");
                    Console.WriteLine($"tokens: {response.Usage?.TotalTokenCount ?? -1}");
                }
                else
                {
                    Console.WriteLine($"Message: {response.Message.Text?.Trim() ?? "?"}");
                }
                Console.WriteLine();
                Console.ResetColor();
            }
        }
    }

    public class RagData
    {
        [VectorStoreRecordKey]
        public int Key { get; set; }
        [VectorStoreRecordData]
        public string Text { get; set; }
        [VectorStoreRecordVector(768, DistanceFunction.CosineSimilarity)]
        public ReadOnlyMemory<float> Vector { get; set; }
    }
}
