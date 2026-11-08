
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

import java.util.List;

public class RagNaif {

    public static void main(String[] args) {

        String apiKey = System.getenv("GEMINI_KEY");

        if (apiKey == null || apiKey.isBlank()) {
            System.out.println("GEMINI_KEY n'est pas définie.");
            return;
        }

        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.2)
                .build();

        // PHASE 1 : INGESTION DU DOCUMENT

        // 1 et 2. Charger le PDF avec Apache Tika
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        System.out.println("Document chargé.");

        // 3. Découper le document en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);

        List<TextSegment> segments = splitter.split(document);

        System.out.println("Nombre de segments : " + segments.size());

        // 4. Créer le modèle d'embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 5. Calculer les embeddings
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();

        System.out.println("Nombre d'embeddings : " + embeddings.size());

        // 6. Enregistrer les embeddings et leurs segments
        EmbeddingStore<TextSegment> embeddingStore =
                new InMemoryEmbeddingStore<>();

        embeddingStore.addAll(embeddings, segments);

        System.out.println("Embeddings enregistrés avec succès.");
    }
}
