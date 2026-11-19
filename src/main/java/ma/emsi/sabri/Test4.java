
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

import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;

import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;

import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;

import dev.langchain4j.service.AiServices;

import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class Test4 {

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

        // 3. Charger le document PDF
        Document document = ClassPathDocumentLoader.loadDocument(
                "rag.pdf",
                new ApacheTikaDocumentParser()
        );

        // 4. Découper le PDF en segments
        List<TextSegment> segments =
                DocumentSplitters.recursive(300, 30).split(document);

        // 5. Créer et remplir l'EmbeddingStore
        EmbeddingStore<TextSegment> store =
                new InMemoryEmbeddingStore<>();

        store.addAll(
                embeddingModel.embedAll(segments).content(),
                segments
        );

        System.out.println(
                "rag.pdf : " + segments.size() + " segments"
        );

        // 6. Créer le ContentRetriever du PDF
        ContentRetriever retrieverPDF =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(embeddingModel)
                        .maxResults(2)
                        .minScore(0.5)
                        .build();

        // 7. Créer le moteur de recherche Tavily
        WebSearchEngine webSearchEngine =
                TavilyWebSearchEngine.builder()
                        .apiKey(System.getenv("TAVILY_API_KEY"))
                        .build();

        // 8. Créer le ContentRetriever du Web
        ContentRetriever retrieverWeb =
                WebSearchContentRetriever.builder()
                        .webSearchEngine(webSearchEngine)
                        .build();

        // ====================================================
        // 9. ANCIEN CODE : utilisation systématique des
        // deux retrievers (PDF + Web).
        //
        // Problème : même pour une question sur le football,
        // des passages sans rapport du PDF sont ajoutés
        // au prompt envoyé au LLM.
        // ====================================================

        /*
        QueryRouter queryRouter =
                new DefaultQueryRouter(
                        retrieverPDF,
                        retrieverWeb
                );
        */

        // ====================================================
        // 10. BONUS : routage intelligent avec Gemini
        // ====================================================

        // Décrire les deux sources pour aider Gemini
        // à choisir celle qui convient à la question.
        Map<ContentRetriever, String> descriptions = Map.of(
                retrieverPDF,
                "Support de cours PDF sur l'intelligence " +
                        "artificielle, le RAG, le fine-tuning et " +
                        "les modèles de langage.",

                retrieverWeb,
                "Recherche sur le Web pour les événements " +
                        "récents, l'actualité, le sport et les " +
                        "informations générales non présentes " +
                        "dans le support de cours."
        );

        // Gemini sélectionne le ou les retrievers pertinents
        QueryRouter queryRouter =
                new LanguageModelQueryRouter(
                        model,
                        descriptions
                );

        // 11. Créer le RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor =
                DefaultRetrievalAugmentor.builder()
                        .queryRouter(queryRouter)
                        .build();

        // 12. Créer l'assistant
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(model)
                        .retrievalAugmentor(retrievalAugmentor)
                        .build();

        // 13. Lancer la conversation interactive
        Scanner scanner = new Scanner(System.in);

        while (true) {

            System.out.println(
                    "========================================"
            );

            System.out.println(
                    "Posez votre question (ou 'fin' pour quitter) :"
            );

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
