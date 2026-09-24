"""
Database models and connection management for LLM Service
==========================================================

Handles PostgreSQL integration for model persistence and analytics.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()

class ModelTrainingSample(Base):
    """Store training samples for model improvement."""
    __tablename__ = "training_samples"

    id = Column(Integer, primary_key=True, index=True)
    secret_value_hash = Column(String(64), index=True)  # SHA-256 hash for privacy
    context_hash = Column(String(64), index=True)      # SHA-256 hash for privacy
    features = Column(JSON)                            # Feature vector
    label = Column(String(20), index=True)             # high/medium/low/false_positive
    user_action = Column(String(50))                   # confirmed_secret/ignored_warning/etc
    confidence_score = Column(Float)                   # Model confidence
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    model_version = Column(String(50), index=True)     # Which model version this was for

class AnalysisRequest(Base):
    """Store analysis request metrics for analytics."""
    __tablename__ = "analysis_requests"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(32), unique=True, index=True)
    client_ip = Column(String(45))                     # IPv4/IPv6 support
    user_agent = Column(Text)
    secret_type = Column(String(50))                   # API Key, Token, etc.
    risk_level = Column(String(20))                    # critical/high/medium/low
    confidence = Column(Float)
    processing_time_ms = Column(Integer)
    cache_hit = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    api_key_hash = Column(String(64), index=True)      # Hashed API key for usage tracking

class ModelVersion(Base):
    """Track model versions and performance metrics."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_name = Column(String(100), unique=True, index=True)
    description = Column(Text)
    accuracy = Column(Float)
    total_predictions = Column(Integer, default=0)
    correct_predictions = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=False)
    parameters = Column(JSON)                          # Model hyperparameters

class SystemMetrics(Base):
    """Store system performance metrics."""
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String(50), index=True)      # cpu_usage, memory_usage, etc.
    value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    service_instance = Column(String(100))            # For multi-instance deployments

class CommunityBlacklist(Base):
    """Promoted hashes — synced to all clients."""
    __tablename__ = "community_blacklist"

    id          = Column(Integer, primary_key=True, index=True)
    hash        = Column(String(16), unique=True, index=True)   # composite hash prefix
    vote_count  = Column(Integer, default=0)
    promoted_at = Column(DateTime, default=datetime.utcnow, index=True)

class StagingQueue(Base):
    """Candidate hashes waiting for consensus before promotion."""
    __tablename__ = "staging_queue"

    id           = Column(Integer, primary_key=True, index=True)
    hash         = Column(String(16), unique=True, index=True)
    machine_ids  = Column(JSON, default=list)   # list of voters
    vote_weight  = Column(Float, default=0.0)
    llm_verified = Column(Boolean, default=False)
    created_at   = Column(DateTime, default=datetime.utcnow, index=True)

class MachineReputation(Base):
    __tablename__ = "machine_reputation"
    machine_id = Column(String(64), primary_key=True, index=True)
    correct    = Column(Integer, default=0)
    wrong      = Column(Integer, default=0)

class FPVotes(Base):
    __tablename__ = "fp_votes"
    id         = Column(Integer, primary_key=True)
    hash       = Column(String(16), index=True)
    machine_id = Column(String(64))

class ModelSnapshot(Base):
    """Persists trained model weights across Railway redeploys."""
    __tablename__ = "model_snapshots"

    id          = Column(Integer, primary_key=True, index=True)
    version     = Column(String(50), unique=True, index=True)
    weights_json = Column(Text)          # JSON-serialised model weights
    is_active   = Column(Boolean, default=False, index=True)
    num_samples = Column(Integer, default=0)   # training samples used
    saved_at    = Column(DateTime, default=datetime.utcnow, index=True)


class ExtensionClient(Base):
    """Registered VS Code extension installations with per-machine credentials."""
    __tablename__ = "extension_clients"

    machine_id        = Column(String(128), primary_key=True, index=True)
    client_secret     = Column(String(64), nullable=False)
    is_active         = Column(Boolean, default=True, index=True)
    vscode_version    = Column(String(50), nullable=True)
    extension_version = Column(String(50), nullable=True)
    client_ip         = Column(String(45), nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow, index=True)
    last_seen_at      = Column(DateTime, default=datetime.utcnow, index=True)

class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///./llm_service.db')
        self.engine = None
        self.SessionLocal = None
        # Don't initialize database on import - do it lazily
        # self._initialize_database()

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Fix legacy Heroku/Railway postgres:// URI scheme for SQLAlchemy
            if self.database_url.startswith('postgres://'):
                self.database_url = self.database_url.replace('postgres://', 'postgresql://', 1)

            # Configure connection pool for production
            connect_args = {}
            if self.database_url.startswith('sqlite'):
                connect_args = {"check_same_thread": False}
            else:
                # Railway internal network does not use SSL; Supabase/Neon requires it.
                # 'prefer' attempts SSL but connects without SSL if server doesn't support it.
                ssl_mode = "require" if ("supabase" in self.database_url or "pooler" in self.database_url) else "prefer"
                connect_args = {
                    "sslmode": ssl_mode,
                    "connect_timeout": 10,
                }

            self.engine = create_engine(
                self.database_url,
                connect_args=connect_args,
                poolclass=QueuePool if not self.database_url.startswith('sqlite') else None,
                pool_pre_ping=True,
                pool_recycle=300,
                pool_size=10,
                max_overflow=20,
                echo=False  # Set to True for debugging
            )

            # Ensure pgvector extension is enabled (for PostgreSQL)
            if not self.database_url.startswith('sqlite'):
                try:
                    with self.engine.connect() as conn:
                        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                        conn.commit()
                        print("✅ pgvector extension verified/enabled")
                except Exception as ex_err:
                    print(f"⚠️ Could not enable pgvector extension: {ex_err}")

            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            # Create tables
            Base.metadata.create_all(bind=self.engine)
            db_type = "SQLite" if self.database_url.startswith('sqlite') else "PostgreSQL"
            print(f"✅ Database ({db_type}) initialized successfully and tables verified")

        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            # Fallback to in-memory SQLite for development
            if not self.database_url.startswith('sqlite'):
                print("🔄 Falling back to SQLite...")
                self.database_url = 'sqlite:///./llm_service.db'
                self._initialize_database()

    def initialize(self):
        """Explicitly initialize engine and verify tables on startup."""
        if self.engine is None or self.SessionLocal is None:
            self._initialize_database()

    def get_session(self) -> Session:
        """Get a database session."""
        if self.engine is None or self.SessionLocal is None:
            self._initialize_database()
        return self.SessionLocal()

    def store_training_sample(self, secret_hash: str, context_hash: str, features: List[float],
                            label: str, user_action: str, confidence: float, model_version: str = "default"):
        """Store a training sample in the database."""
        try:
            with self.get_session() as session:
                sample = ModelTrainingSample(
                    secret_value_hash=secret_hash,
                    context_hash=context_hash,
                    features=features,
                    label=label,
                    user_action=user_action,
                    confidence_score=confidence,
                    model_version=model_version
                )
                session.add(sample)
                session.commit()
                return sample.id
        except SQLAlchemyError as e:
            print(f"❌ Failed to store training sample: {e}")
            return None

    def store_analysis_request(self, request_id: str, client_ip: str, user_agent: str,
                             secret_type: str, risk_level: str, confidence: float,
                             processing_time_ms: int, cache_hit: bool, api_key_hash: str):
        """Store analysis request metrics."""
        try:
            with self.get_session() as session:
                request = AnalysisRequest(
                    request_id=request_id,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    secret_type=secret_type,
                    risk_level=risk_level,
                    confidence=confidence,
                    processing_time_ms=processing_time_ms,
                    cache_hit=cache_hit,
                    api_key_hash=api_key_hash
                )
                session.add(request)
                session.commit()
                return request.id
        except SQLAlchemyError as e:
            print(f"❌ Failed to store analysis request: {e}")
            return None

    def get_training_samples(self, limit: int = 1000, model_version: str = None) -> List[Dict[str, Any]]:
        """Retrieve training samples for model training."""
        try:
            with self.get_session() as session:
                query = session.query(ModelTrainingSample)
                if model_version:
                    query = query.filter(ModelTrainingSample.model_version == model_version)
                samples = query.order_by(ModelTrainingSample.created_at.desc()).limit(limit).all()

                return [{
                    'id': s.id,
                    'secret_hash': s.secret_value_hash,
                    'context_hash': s.context_hash,
                    'features': s.features,
                    'label': s.label,
                    'user_action': s.user_action,
                    'confidence': s.confidence_score,
                    'created_at': s.created_at.isoformat(),
                    'model_version': s.model_version
                } for s in samples]
        except SQLAlchemyError as e:
            print(f"❌ Failed to retrieve training samples: {e}")
            return []

    def get_analytics_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get analytics summary for the specified number of days."""
        try:
            with self.get_session() as session:
                # Calculate date threshold
                threshold = datetime.utcnow() - timedelta(days=days)

                # Query metrics
                total_requests = session.query(AnalysisRequest).filter(
                    AnalysisRequest.created_at >= threshold
                ).count()

                cache_hits = session.query(AnalysisRequest).filter(
                    AnalysisRequest.created_at >= threshold,
                    AnalysisRequest.cache_hit == True
                ).count()

                avg_processing_time = session.query(AnalysisRequest).filter(
                    AnalysisRequest.created_at >= threshold
                ).with_entities(AnalysisRequest.processing_time_ms).all()

                processing_times = [t[0] for t in avg_processing_time if t[0] is not None]
                avg_time = sum(processing_times) / len(processing_times) if processing_times else 0

                # Risk level distribution
                risk_counts = {}
                risk_results = session.query(AnalysisRequest.risk_level, AnalysisRequest.id).filter(
                    AnalysisRequest.created_at >= threshold
                ).all()

                for risk_level, _ in risk_results:
                    risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1

                return {
                    'total_requests': total_requests,
                    'cache_hit_rate': cache_hits / total_requests if total_requests > 0 else 0,
                    'average_processing_time_ms': avg_time,
                    'risk_distribution': risk_counts,
                    'period_days': days
                }
        except SQLAlchemyError as e:
            print(f"❌ Failed to get analytics summary: {e}")
            return {}

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data to manage database size."""
        try:
            with self.get_session() as session:
                threshold = datetime.utcnow() - timedelta(days=days_to_keep)

                # Delete old training samples (keep recent ones)
                deleted_samples = session.query(ModelTrainingSample).filter(
                    ModelTrainingSample.created_at < threshold
                ).delete()

                # Delete old analysis requests
                deleted_requests = session.query(AnalysisRequest).filter(
                    AnalysisRequest.created_at < threshold
                ).delete()

                # Delete old system metrics
                deleted_metrics = session.query(SystemMetrics).filter(
                    SystemMetrics.timestamp < threshold
                ).delete()

                session.commit()

                print(f"🧹 Cleaned up {deleted_samples} training samples, {deleted_requests} requests, {deleted_metrics} metrics")
                return True
        except SQLAlchemyError as e:
            print(f"❌ Failed to cleanup old data: {e}")
            return False

    def optimize_database(self):
        """Optimize database performance with indexes and maintenance."""
        try:
            with self.engine.connect() as conn:
                # Create indexes for better query performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_requests(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_analysis_client_ip ON analysis_requests(client_ip)",
                    "CREATE INDEX IF NOT EXISTS idx_analysis_risk_level ON analysis_requests(risk_level)",
                    "CREATE INDEX IF NOT EXISTS idx_analysis_secret_type ON analysis_requests(secret_type)",
                    "CREATE INDEX IF NOT EXISTS idx_analysis_cache_hit ON analysis_requests(cache_hit)",
                    "CREATE INDEX IF NOT EXISTS idx_training_secret_hash ON training_samples(secret_value_hash)",
                    "CREATE INDEX IF NOT EXISTS idx_training_model_version ON training_samples(model_version)",
                    "CREATE INDEX IF NOT EXISTS idx_training_timestamp ON training_samples(created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_training_label ON training_samples(label)",
                    "CREATE INDEX IF NOT EXISTS idx_blacklist_hash     ON community_blacklist(hash)",
                    "CREATE INDEX IF NOT EXISTS idx_blacklist_promoted ON community_blacklist(promoted_at)",
                    "CREATE INDEX IF NOT EXISTS idx_staging_hash       ON staging_queue(hash)",
                    "CREATE INDEX IF NOT EXISTS idx_staging_weight     ON staging_queue(vote_weight)",
                ]

                for index_sql in indexes:
                    try:
                        conn.execute(text(index_sql))
                        conn.commit()
                    except Exception as idx_error:
                        print(f"Index creation failed: {idx_error}")

                # Analyze tables for query optimization (PostgreSQL)
                if not self.database_url.startswith('sqlite'):
                    try:
                        conn.execute(text("ANALYZE analysis_requests"))
                        conn.execute(text("ANALYZE training_samples"))
                        conn.commit()
                        print("Database analysis completed")
                    except Exception as analyze_error:
                        print(f"Table analysis failed: {analyze_error}")

                # Vacuum database (SQLite specific optimization)
                if self.database_url.startswith('sqlite'):
                    try:
                        conn.execute(text("VACUUM"))
                        conn.commit()
                        print("Database vacuum completed")
                    except Exception as vacuum_error:
                        print(f"Vacuum failed: {vacuum_error}")

        except Exception as e:
            print(f"Database optimization failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics."""
        try:
            stats = {
                'database_type': 'postgresql' if not self.database_url.startswith('sqlite') else 'sqlite',
                'connection_pool': {
                    'pool_size': getattr(self.engine.pool, 'size', 'unknown'),
                    'checked_out': getattr(self.engine.pool, 'checkedout', lambda: 'unknown')() if hasattr(self.engine.pool, 'checkedout') else 'unknown',
                },
                'last_optimized': datetime.utcnow().isoformat()
            }

            # Get table statistics
            with self.get_session() as session:
                # Analysis requests count
                analysis_count = session.query(AnalysisRequest).count()
                training_count = session.query(ModelTrainingSample).count()

                stats['tables'] = {
                    'analysis_requests': {'row_count': analysis_count},
                    'training_samples': {'row_count': training_count}
                }

            return stats

        except Exception as e:
            print(f"Failed to get performance stats: {e}")
            return {'error': str(e)}

    # ─── Community Blacklist Methods ──────────────────────────────────────────

    def submit_hash(self, hash_val: str, machine_id: str, weight: float) -> str:
        """Add a vote for a hash. Returns: 'promoted' | 'staged' | 'duplicate'."""
        try:
            with self.get_session() as session:
                # already in blacklist?
                if session.query(CommunityBlacklist).filter_by(hash=hash_val).first():
                    return 'promoted'

                entry = session.query(StagingQueue).filter_by(hash=hash_val).first()

                if not entry:
                    entry = StagingQueue(hash=hash_val, machine_ids=[], vote_weight=0.0)
                    session.add(entry)

                if machine_id in entry.machine_ids:
                    return 'duplicate'

                # Update machine_ids list (handling list immutability in some SQLALchemy versions)
                updated_ids = list(entry.machine_ids)
                updated_ids.append(machine_id)
                entry.machine_ids = updated_ids
                entry.vote_weight = entry.vote_weight + weight

                session.commit()

                if entry.vote_weight >= 3.0 and entry.llm_verified:
                    return self._promote(session, hash_val)

                return 'staged'
        except SQLAlchemyError as e:
            print(f"❌ submit_hash failed: {e}")
            return 'error'

    def mark_llm_verified(self, hash_val: str) -> bool:
        """Server verified the hash with its own LLM. Try to promote."""
        try:
            with self.get_session() as session:
                entry = session.query(StagingQueue).filter_by(hash=hash_val).first()
                if not entry:
                    return False
                entry.llm_verified = True
                session.commit()
                if entry.vote_weight >= 3.0:
                    self._promote(session, hash_val)
                return True
        except SQLAlchemyError as e:
            print(f"❌ mark_llm_verified failed: {e}")
            return False

    def _promote(self, session: Session, hash_val: str) -> str:
        promoted = CommunityBlacklist(hash=hash_val)
        session.add(promoted)
        session.query(StagingQueue).filter_by(hash=hash_val).delete()
        session.commit()
        return 'promoted'

    def get_blacklist_hashes(self) -> list[str]:
        """Return all promoted hashes for client sync."""
        try:
            with self.get_session() as session:
                rows = session.query(CommunityBlacklist.hash).all()
                return [r[0] for r in rows]
        except SQLAlchemyError as e:
            print(f"❌ get_blacklist_hashes failed: {e}")
            return []

    def remove_false_positive(self, hash_val: str) -> bool:
        """Remove a hash that reached FP consensus."""
        try:
            with self.get_session() as session:
                session.query(CommunityBlacklist).filter_by(hash=hash_val).delete()
                session.commit()
                return True
        except SQLAlchemyError as e:
            print(f"❌ remove_false_positive failed: {e}")
            return False

    def get_machine_reputation(self, machine_id: str) -> dict:
        try:
            with self.get_session() as session:
                r = session.query(MachineReputation).filter_by(machine_id=machine_id).first()
                return {"correct": r.correct, "wrong": r.wrong} if r else {"correct": 0, "wrong": 0}
        except SQLAlchemyError:
            return {"correct": 0, "wrong": 0}

    def penalize_machine(self, machine_id: str, correct: bool):
        try:
            with self.get_session() as session:
                r = session.query(MachineReputation).filter_by(machine_id=machine_id).first()
                if not r:
                    r = MachineReputation(machine_id=machine_id)
                    session.add(r)
                if correct: r.correct += 1
                else:       r.wrong   += 1
                session.commit()
        except SQLAlchemyError as e:
            print(f"❌ penalize_machine failed: {e}")

    def increment_fp_votes(self, hash_val: str, machine_id: str) -> int:
        try:
            with self.get_session() as session:
                exists = session.query(FPVotes).filter_by(hash=hash_val, machine_id=machine_id).first()
                if not exists:
                    session.add(FPVotes(hash=hash_val, machine_id=machine_id))
                    session.commit()
                return session.query(FPVotes).filter_by(hash=hash_val).count()
        except SQLAlchemyError as e:
            print(f"❌ increment_fp_votes failed: {e}")
            return 0

    def get_staging_entry(self, hash_val: str) -> dict | None:
        try:
            with self.get_session() as session:
                e = session.query(StagingQueue).filter_by(hash=hash_val).first()
                if not e: return None
                return {"vote_weight": e.vote_weight, "llm_verified": e.llm_verified}
        except SQLAlchemyError:
            return None

    def count_training_samples(self) -> int:
        try:
            with self.get_session() as session:
                return session.query(ModelTrainingSample).count()
        except SQLAlchemyError:
            return 0

    # ── Model Persistence (T05) ───────────────────────────────────────────────

    def save_model_to_db(self, weights_json: str, version: str = "active",
                         num_samples: int = 0) -> bool:
        """
        Persist model weights JSON in the database.
        Overwrites the existing 'active' snapshot so there's always one record.
        Survives Railway redeploys because the DB is an external service.
        """
        try:
            with self.get_session() as session:
                snap = session.query(ModelSnapshot).filter_by(version=version).first()
                if snap:
                    snap.weights_json = weights_json
                    snap.num_samples  = num_samples
                    snap.saved_at     = datetime.utcnow()
                    snap.is_active    = True
                else:
                    # Mark all others inactive first
                    session.query(ModelSnapshot).update({"is_active": False})
                    snap = ModelSnapshot(
                        version=version,
                        weights_json=weights_json,
                        is_active=True,
                        num_samples=num_samples,
                    )
                    session.add(snap)
                session.commit()
                print(f"💾 Model snapshot '{version}' saved to DB ({num_samples} samples)")
                return True
        except SQLAlchemyError as e:
            print(f"❌ save_model_to_db failed: {e}")
            return False

    def load_model_from_db(self, version: str = "active") -> str | None:
        """
        Load the most recent active model weights JSON from DB.
        Returns the raw JSON string, or None if no snapshot exists yet.
        """
        try:
            with self.get_session() as session:
                snap = session.query(ModelSnapshot).filter_by(
                    version=version, is_active=True
                ).order_by(ModelSnapshot.saved_at.desc()).first()
                if snap:
                    print(f"✅ Model snapshot '{version}' loaded from DB (saved {snap.saved_at})")
                    return snap.weights_json
                return None
        except SQLAlchemyError as e:
            print(f"❌ load_model_from_db failed: {e}")
            return None

    # ── Extension Client Handshake & Auth ──────────────────────────────────────

    def register_or_get_client(self, machine_id: str, vscode_version: str = "",
                               extension_version: str = "", client_ip: str = "") -> tuple[str, bool, datetime]:
        """
        Idempotent client registration.
        If machine_id is already registered, updates metadata and returns (client_secret, is_new=False, created_at).
        If not registered, generates a secure unique secret, persists it, and returns (client_secret, is_new=True, created_at).
        """
        import secrets as py_secrets
        try:
            with self.get_session() as session:
                client = session.query(ExtensionClient).filter_by(machine_id=machine_id).first()
                new_secret = "sec_" + py_secrets.token_hex(28)
                now = datetime.utcnow()

                if client:
                    client.client_secret = new_secret
                    client.last_seen_at = now
                    client.is_active = True
                    if vscode_version:
                        client.vscode_version = vscode_version
                    if extension_version:
                        client.extension_version = extension_version
                    if client_ip:
                        client.client_ip = client_ip
                    session.commit()
                    print(f"🔄 Rotated credentials for existing machine: {machine_id[:12]}...")
                    return new_secret, False, client.created_at

                client = ExtensionClient(
                    machine_id=machine_id,
                    client_secret=new_secret,
                    is_active=True,
                    vscode_version=vscode_version,
                    extension_version=extension_version,
                    client_ip=client_ip,
                    created_at=now,
                    last_seen_at=now
                )
                session.add(client)
                session.commit()
                print(f"🔑 Registered new extension client for machine: {machine_id[:12]}...")
                return new_secret, True, now
        except SQLAlchemyError as e:
            print(f"❌ register_or_get_client failed: {e}")
            raise

    def get_client_record(self, machine_id: str) -> dict | None:
        """Fetch client credentials and active status by machine_id."""
        try:
            with self.get_session() as session:
                client = session.query(ExtensionClient).filter_by(machine_id=machine_id).first()
                if not client:
                    return None
                return {
                    "machine_id": client.machine_id,
                    "client_secret": client.client_secret,
                    "is_active": client.is_active,
                    "created_at": client.created_at,
                    "last_seen_at": client.last_seen_at,
                }
        except SQLAlchemyError as e:
            print(f"❌ get_client_record failed: {e}")
            return None

    def update_client_last_seen(self, machine_id: str) -> None:
        """Update last_seen_at for an active client without locking."""
        try:
            with self.get_session() as session:
                client = session.query(ExtensionClient).filter_by(machine_id=machine_id).first()
                if client:
                    client.last_seen_at = datetime.utcnow()
                    session.commit()
        except SQLAlchemyError:
            pass


# Global database manager instance
db_manager = DatabaseManager()
