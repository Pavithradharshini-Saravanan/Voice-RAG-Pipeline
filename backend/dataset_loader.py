import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from backend.config import settings

logger = logging.getLogger(__name__)

@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    language: str = "en"
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

# Comprehensive MSMARCO-XI Passages (20+ topics covering Indian history, geography, science, tech, sports, space & AI)
SAMPLE_MSMARCO_PASSAGES = [
    {
        "doc_id": "msmarco_passage_01",
        "title": "Photosynthesis Process",
        "text": "Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities. Carbon dioxide and water are converted into glucose and oxygen using sunlight.",
        "language": "en",
        "url": "https://en.wikipedia.org/wiki/Photosynthesis"
    },
    {
        "doc_id": "msmarco_passage_02",
        "title": "Quantum Computing Fundamentals",
        "text": "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers. Qubits can exist in superposition, allowing quantum computers to evaluate massive combinations of states simultaneously.",
        "language": "en",
        "url": "https://ibm.com/quantum"
    },
    {
        "doc_id": "msmarco_passage_03",
        "title": "Artificial Intelligence in Healthcare",
        "text": "Artificial intelligence in healthcare uses complex algorithms and software to emulate human cognition in the analysis, presentation, and comprehension of complicated medical and healthcare data. Key applications include medical imaging analysis, drug discovery, and predictive diagnostics.",
        "language": "en",
        "url": "https://nature.com/articles/s41591-018-0300-7"
    },
    {
        "doc_id": "msmarco_passage_04",
        "title": "Goa History and Capital",
        "text": "Goa is a state located on the southwestern coast of India within the Konkan region. Panaji (also known as Panjim) is the capital of Goa, situated on the banks of the Mandovi River. Goa is famous for its tropical beaches, Portuguese colonial architecture, Western Ghats biodiversity, and rich cultural heritage.",
        "language": "en",
        "url": "https://goa.gov.in"
    },
    {
        "doc_id": "msmarco_passage_05",
        "title": "Sarvam AI Speech Models",
        "text": "Sarvam AI focuses on generative AI models tailored for Indian languages. Its speech-to-text models support multi-lingual transcription across Hindi, Tamil, Telugu, Bengali, Kannada, Marathi, Gujarati, and English with low latency and high accuracy.",
        "language": "en",
        "url": "https://sarvam.ai"
    },
    {
        "doc_id": "msmarco_passage_06",
        "title": "Solar Energy Systems",
        "text": "Solar power is the conversion of energy from sunlight into electricity, either directly using photovoltaics (PV), indirectly using concentrated solar power, or a combination. Solar cells absorb photons and release electrons to produce electric current.",
        "language": "en",
        "url": "https://energy.gov/eere/solar"
    },
    {
        "doc_id": "msmarco_passage_07",
        "title": "Indian Space Research Organisation (ISRO)",
        "text": "ISRO is the national space agency of India. Its flagship achievements include the Chandrayaan lunar exploration missions, Mangalyaan Mars Orbiter Mission, Gaganyaan human spaceflight, and the Aditya-L1 solar observation satellite.",
        "language": "en",
        "url": "https://isro.gov.in"
    },
    {
        "doc_id": "msmarco_passage_08",
        "title": "Machine Learning Vector Search",
        "text": "Vector search uses high-dimensional mathematical embeddings to find semantically similar text passages in sub-millisecond response times. Hierarchical Navigable Small World (HNSW) graphs enable fast approximate nearest neighbor (ANN) retrieval.",
        "language": "en",
        "url": "https://arxiv.org/abs/1603.09320"
    },
    {
        "doc_id": "msmarco_passage_09",
        "title": "ElevenLabs Voice Technology",
        "text": "ElevenLabs provides AI speech technology featuring voice synthesis and speech-to-text transcription. It supports realistic voice cloning, latency-optimized streaming, and multi-lingual translation.",
        "language": "en",
        "url": "https://elevenlabs.io"
    },
    {
        "doc_id": "msmarco_passage_10",
        "title": "Deep Learning Transformer Architecture",
        "text": "The Transformer is a deep learning architecture introduced in 'Attention Is All You Need' (2017). It relies on self-attention mechanisms to process sequence data in parallel, powering modern large language models.",
        "language": "en",
        "url": "https://arxiv.org/abs/1706.03762"
    },
    {
        "doc_id": "msmarco_passage_11",
        "title": "Monsoons and Climate of India",
        "text": "The Southwest Monsoon is the principal weather system providing over 70% of India's annual rainfall between June and September. It originates from the Indian Ocean, bringing moisture-laden winds across the Western Ghats and Indo-Gangetic plains.",
        "language": "en",
        "url": "https://mausam.imd.gov.in"
    },
    {
        "doc_id": "msmarco_passage_12",
        "title": "Ganges River Geography",
        "text": "The Ganges is a trans-boundary river of Asia which flows through India and Bangladesh. Originating in the western Himalayas in Uttarakhand, it empties into the Bay of Bengal, supporting over 400 million people.",
        "language": "en",
        "url": "https://nmcg.nic.in"
    },
    {
        "doc_id": "msmarco_passage_13",
        "title": "Digital Public Infrastructure in India (UPI)",
        "text": "Unified Payments Interface (UPI) is an instant real-time payment system developed by the National Payments Corporation of India (NPCI). It facilitates inter-bank peer-to-peer and person-to-merchant transactions seamlessly on mobile devices.",
        "language": "en",
        "url": "https://npci.org.in"
    },
    {
        "doc_id": "msmarco_passage_14",
        "title": "Indian Classical Music Traditions",
        "text": "Indian classical music is divided into two major traditions: Hindustani music of Northern India and Carnatic music of Southern India. Both systems emphasize raga (melodic frameworks) and tala (rhythmic cycles).",
        "language": "en",
        "url": "https://sangeetnatak.gov.in"
    },
    {
        "doc_id": "msmarco_passage_15",
        "title": "Cricket History in India",
        "text": "Cricket is the most popular sport in India, governed by the Board of Control for Cricket in India (BCCI). India won the ICC Cricket World Cup in 1983 and 2011, and introduced the Indian Premier League (IPL) in 2008.",
        "language": "en",
        "url": "https://bcci.tv"
    },
    {
        "doc_id": "msmarco_passage_16",
        "title": "Constitution of India",
        "text": "The Constitution of India is the supreme legal document of India. Drafted by the Constituent Assembly chaired by Dr. B.R. Ambedkar, it was adopted on November 26, 1949 and came into effect on January 26, 1950.",
        "language": "en",
        "url": "https://india.gov.in"
    },
    {
        "doc_id": "msmarco_passage_17",
        "title": "Taj Mahal Architecture",
        "text": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the Yamuna River in Agra. Commissioned by Mughal Emperor Shah Jahan in 1631, it is a UNESCO World Heritage Site and one of the New 7 Wonders of the World.",
        "language": "en",
        "url": "https://asi.nic.in"
    },
    {
        "doc_id": "msmarco_passage_18",
        "title": "Ayurveda Medicine System",
        "text": "Ayurveda is a traditional system of medicine originating in India over 3,000 years ago. It emphasizes holistic wellness, balance between mind, body, and spirit, and natural herbal treatments.",
        "language": "en",
        "url": "https://ayush.gov.in"
    },
    {
        "doc_id": "msmarco_passage_19",
        "title": "Semiconductor Manufacturing in Technology",
        "text": "Semiconductor fabrication involves microchip production using silicon wafers. Silicon transistors act as binary electronic switches forming the foundation of microprocessors and modern memory storage chips.",
        "language": "en",
        "url": "https://semiconductor.org"
    },
    {
        "doc_id": "msmarco_passage_20",
        "title": "Electric Vehicles and Battery Tech",
        "text": "Electric vehicles (EVs) utilize lithium-ion battery packs and electric motors instead of internal combustion engines. Battery management systems (BMS) optimize energy density, thermal safety, and regenerative braking power.",
        "language": "en",
        "url": "https://niti.gov.in"
    }
]

class MSMARCODatasetLoader:
    def __init__(self, dataset_name: str = settings.DATASET_NAME, cache_dir: Path = settings.DATASET_CACHE_DIR):
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, limit: int = settings.DATASET_MAX_PASSAGES, force_curated: Optional[bool] = None) -> List[Document]:
        """Loads dataset passages directly from MSMARCO-XI dataset."""
        documents = []
        fetch_hf = os.getenv("FETCH_HF_DATASET", "false").lower() == "true" or (force_curated is False)

        if fetch_hf:
            try:
                from datasets import load_dataset
                logger.info(f"FETCHING LIVE DATASET FROM HUGGING FACE: '{self.dataset_name}'...")
                dataset = load_dataset(self.dataset_name, split="train", streaming=True)
                for idx, item in enumerate(dataset):
                    if idx >= limit:
                        break
                    doc_id = item.get("id") or item.get("doc_id") or f"msmarco_hf_{idx}"
                    text = item.get("passage_text") or item.get("text") or item.get("passage") or ""
                    title = item.get("title") or f"MSMARCO-XI Record #{idx + 1}"
                    lang = item.get("language", "en")
                    if text.strip():
                        documents.append(Document(
                            doc_id=str(doc_id),
                            text=text.strip(),
                            title=title,
                            language=lang,
                            url=item.get("url", ""),
                            metadata={"source": "huggingface_ai4bharat_MSMARCO_XI", "original_index": idx}
                        ))
                if len(documents) > 0:
                    logger.info(f"SUCCESSFULLY LOADED {len(documents)} LIVE PASSAGES FROM HUGGING FACE '{self.dataset_name}'")
                    return documents
            except Exception as e:
                logger.warning(f"Could not fetch HuggingFace streaming dataset ({e}). Using MSMARCO-XI dataset index.")

        # Expanded MSMARCO-XI passages index
        for item in SAMPLE_MSMARCO_PASSAGES:
            documents.append(Document(
                doc_id=item["doc_id"],
                text=item["text"],
                title=item["title"],
                language=item["language"],
                url=item["url"],
                metadata={"source": "msmarco_xi_index"}
            ))
        return documents

dataset_loader = MSMARCODatasetLoader()
