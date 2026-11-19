
package ma.emsi.sabri;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TestRoutage {

    public static void main(String[] args) {

        // 1. Créer le modèle d'embedding
        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        // 2. Créer deux EmbeddingStores indépendants
        EmbeddingStore<TextSegment> storeIA =
                creerEmbeddingStore("rag.pdf", embeddingModel);

        EmbeddingStore<TextSegment> storeGeographie =
                creerEmbeddingStore("geographie.txt", embeddingModel);

        System.out.println("Deux EmbeddingStores créés avec succès.");

        // 3. Créer les deux ContentRetrievers
        ContentRetriever retrieverIA =
                creerContentRetriever(storeIA, embeddingModel);

        ContentRetriever retrieverGeographie =
                creerContentRetriever(storeGeographie, embeddingModel);

        System.out.println("Deux ContentRetrievers créés avec succès.");

        // 4. Créer le modèle Gemini
        GoogleGenAiChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        // 5. Associer chaque retriever à sa description
        Map<ContentRetriever, String> descriptions = Map.of(
                retrieverIA,
                "Support de cours sur l'intelligence artificielle, " +
                        "le fine-tuning, le RAG et les modèles de langage.",

                retrieverGeographie,
                "Document de géographie sur le Maroc, " +
                        "ses villes, ses montagnes et sa localisation."
        );

        // 6. Créer le routeur utilisant Gemini
        QueryRouter queryRouter =
                new LanguageModelQueryRouter(model, descriptions);

        System.out.println("LanguageModelQueryRouter créé avec succès.");

        // 7. Créer le RetrievalAugmentor avec le routeur
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // 8. Créer l'assistant avec le RAG routé
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // 9. Lancer la conversation interactive
        Scanner scanner = new Scanner(System.in);

        while (true) {

            System.out.println("==================================================");
            System.out.println("Posez votre question (ou 'fin' pour quitter) :");

            String question = scanner.nextLine();

            if (question.equalsIgnoreCase("fin")) {
                break;
            }

            String reponse = assistant.chat(question);

            System.out.println("Assistant : " + reponse);
        }

        scanner.close();
    }

    // Charger, découper et indexer un document
    private static EmbeddingStore<TextSegment> creerEmbeddingStore(
            String fichier,
            EmbeddingModel embeddingModel) {

        Document document = ClassPathDocumentLoader.loadDocument(
                fichier,
                new ApacheTikaDocumentParser()
        );

        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30).split(document);

        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(
                embeddingModel.embedAll(segments).content(),
                segments
        );

        System.out.println(
                fichier + " : " + segments.size() + " segments"
        );

        return store;
    }

    // Créer un ContentRetriever pour un EmbeddingStore
    private static ContentRetriever creerContentRetriever(
            EmbeddingStore<TextSegment> store,
            EmbeddingModel embeddingModel) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }
}
