
package ma.emsi.sabri;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.google.genai.GoogleGenAiChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.input.PromptTemplate;

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Collections;

public class Test3 {

    public static void main(String[] args) {

        // 1. Créer le modèle Gemini
        ChatModel model = GoogleGenAiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-3.5-flash-lite")
                .temperature(0.2)
                .logRequestsAndResponses(true)
                .build();

        // 2. Créer le modèle d'embedding
        EmbeddingModel embeddingModel =
                new AllMiniLmL6V2EmbeddingModel();

        // 3. Charger uniquement le support de cours RAG
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        // 4. Découper le document en segments
        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30).split(document);

        // 5. Créer et remplir l'EmbeddingStore
        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(
                embeddingModel.embedAll(segments).content(),
                segments
        );

        System.out.println("rag.pdf : " + segments.size() + " segments");

        // 6. Créer le ContentRetriever
        ContentRetriever contentRetriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 7. Créer le PromptTemplate (bonus)
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{question}}' porte sur " +
                        "l'intelligence artificielle, le RAG ou le fine-tuning ? " +
                        "Réponds seulement par 'oui', 'non' ou 'peut-être'."
        );

        // 8. Créer un QueryRouter personnalisé
        class QueryRouterPourEviterRag implements QueryRouter {

            @Override
            public List<ContentRetriever> route(Query query) {

                // Construire le prompt avec la question
                String prompt = promptTemplate.apply(
                        Map.of("question", query.text())
                ).text();

                // Demander directement à Gemini si le RAG est nécessaire
                String reponse = model.chat(prompt);

                System.out.println("Décision du routeur : " + reponse);

                // Si la question ne concerne pas l'IA : pas de RAG
                if (reponse.trim().equalsIgnoreCase("non")) {
                    System.out.println("RAG désactivé.");
                    return Collections.emptyList();
                }

                // Sinon, utiliser le document RAG
                System.out.println("RAG activé.");
                return Collections.singletonList(contentRetriever);
            }
        }

        // 9. Instancier le QueryRouter personnalisé
        QueryRouter queryRouter = new QueryRouterPourEviterRag();

        // 10. Créer le RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // 11. Créer l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // 12. Lancer la conversation interactive
        Scanner scanner = new Scanner(System.in);

        while (true) {

            System.out.println("========================================");
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
}
