
package ma.emsi.sabri;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;

import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Scanner;
import java.util.logging.Logger;
import java.util.logging.Level;
import java.util.logging.ConsoleHandler;

public class RagNaif {

    public static void main(String[] args) {

        // Configuration du logging
        configureLogger();

        // Configuration du modèle Gemini
        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            System.out.println("GEMINI_KEY n'est pas définie.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-3.8-flash")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        // ==========================================
        // PHASE 1 : INGESTION
        // ==========================================

        // 1. Charger le document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("Document chargé.");

        // 2. Découper le document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // 3. Créer le modèle d'embeddings
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 4. Générer les embeddings
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        // 5. Enregistrer les embeddings
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings enregistrés avec succès.");

        // ==========================================
        // PHASE 2 : RETRIEVAL ET GENERATION
        // ==========================================

        // 6. Créer le ContentRetriever
        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(embeddingStore)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 7. Créer la mémoire conversationnelle
        ChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);

        // 8. Créer l'assistant RAG
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(memory)
                .contentRetriever(contentRetriever)
                .build();

        // 9. Démarrer la conversation
        conversationAvec(assistant);
    }

    // ==========================================
    // CONFIGURATION DU LOGGING
    // ==========================================

    private static void configureLogger() {

        Logger packageLogger = Logger.getLogger("dev.langchain4j");

        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();

        handler.setLevel(Level.FINE);

        packageLogger.addHandler(handler);
    }

    // ==========================================
    // CONVERSATION INTERACTIVE
    // ==========================================

    private static void conversationAvec(Assistant assistant) {

        try (Scanner scanner = new Scanner(System.in)) {

            while (true) {

                System.out.println("==================================================");
                System.out.println("Posez votre question : ");

                String question = scanner.nextLine();

                if (question.isBlank()) {
                    continue;
                }

                System.out.println("==================================================");

                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }

                String reponse = assistant.chat(question);

                System.out.println("Assistant : " + reponse);

                System.out.println("==================================================");
            }
        }
    }
}
