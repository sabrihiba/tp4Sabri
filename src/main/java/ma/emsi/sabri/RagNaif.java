
package ma.emsi.sabri;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.query.transformer.CompressingQueryTransformer;
import dev.langchain4j.rag.query.transformer.QueryTransformer;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Scanner;

public class RagNaif {

    interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) {

        // 1. Charger le document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("Document chargé.");

        // 2. Découper le document en segments
        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30).split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // 3. Créer le modèle d'embeddings
        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        // 4. Générer les embeddings
        List<Embedding> embeddings =
                embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        // 5. Enregistrer les embeddings
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings enregistrés avec succès.");

        // 6. Configurer Gemini et activer le logging
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        // 7. Configurer la recherche des segments
        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 8. Transformer les questions selon l'historique
        QueryTransformer queryTransformer =
                new CompressingQueryTransformer(model);

        // 9. Construire le pipeline RAG
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryTransformer(queryTransformer)
                        .contentRetriever(contentRetriever)
                        .build();

        // 10. Créer la mémoire de conversation
        ChatMemory memory =
                MessageWindowChatMemory.withMaxMessages(10);

        // 11. Créer l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // 12. Démarrer la conversation
        conversationAvec(assistant);
    }

    private static void conversationAvec(Assistant assistant) {

        Scanner scanner = new Scanner(System.in);

        while (true) {

            System.out.println("==================================================");
            System.out.println("Posez votre question (ou 'fin' pour quitter) :");

            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("fin")) {
                break;
            }

            System.out.println("==================================================");

            String reponse = assistant.chat(question);

            System.out.println("Assistant : " + reponse);
        }

        scanner.close();
    }
}
