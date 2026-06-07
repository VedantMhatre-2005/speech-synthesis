"""
Knowledge Graph for CP patient (Aarav) — Neo4j backend.

Replaces NetworkX with Neo4j so the graph is:
  • Persisted in a real graph database
  • Fully visualisable in Neo4j Browser (http://localhost:7474)
  • Queryable via Cypher

Run this file directly to (re)build the graph:
    python knowledge_graph.py
"""

from neo4j import GraphDatabase

# ── CONNECTION CONFIG ─────────────────────────────────────────
NEO4J_URI      = "neo4j://127.0.0.1:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password"          # change to match your Neo4j Desktop password


# ── DRIVER SINGLETON ──────────────────────────────────────────

class Neo4jKG:
    """
    Thin wrapper around the Neo4j driver.
    Exposes build(), query_context(), and format_context_for_prompt().
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        # Verify connection immediately
        self.driver.verify_connectivity()
        print("[Neo4j] Connected successfully.")

    def close(self):
        self.driver.close()

    def run(self, cypher: str, **params):
        """Execute a Cypher statement and return all records."""
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return result.data()

    # ── BUILD ─────────────────────────────────────────────────

    def clear(self):
        """Wipe the entire database before rebuilding."""
        self.run("MATCH (n) DETACH DELETE n")
        print("[Neo4j] Database cleared.")

    def build(self):
        """
        Populates Neo4j with:
          • 1  Patient node
          • 5  Person nodes
          • 3  Routine nodes
          • 10 Vocabulary nodes
          • 5  Episode nodes
        And all relationships between them.
        """
        self.clear()

        # ── PATIENT ───────────────────────────────────────────
        self.run("""
            CREATE (:Patient {
                name:           'Aarav',
                age:            8,
                cp_subtype:     'spastic_diplegia',
                linguistic_age: 6.2,
                mcds_baseline:  0.58,
                aac_device:     'Tobii Dynavox I-13'
            })
        """)

        # ── PEOPLE ────────────────────────────────────────────
        people = [
            dict(name="Priya",      role="mother",
                 nickname="mama",   contact_frequency="daily",
                 communication_style="simple_sentences"),
            dict(name="Vijay",      role="father",
                 nickname="papa",   contact_frequency="daily",
                 communication_style="simple_sentences"),
            dict(name="Rohan",      role="classmate",
                 nickname="Rohan",  contact_frequency="weekdays",
                 communication_style="peer",
                 shared_interests="dinosaurs,football"),
            dict(name="Dr. Meera",  role="speech_therapist",
                 nickname="Dr. Meera", contact_frequency="weekly",
                 communication_style="structured"),
            dict(name="Dr. Sharma", role="physiotherapist",
                 nickname="Dr. Sharma", contact_frequency="twice_weekly",
                 communication_style="clinical"),
        ]
        for p in people:
            self.run("""
                CREATE (:Person {
                    name:                 $name,
                    role:                 $role,
                    nickname:             $nickname,
                    contact_frequency:    $contact_frequency,
                    communication_style:  $communication_style
                })
            """, **{k: p.get(k, "") for k in
                    ["name","role","nickname","contact_frequency","communication_style"]})

        # ── PATIENT → PEOPLE RELATIONSHIPS ────────────────────
        relationships = [
            ("Aarav", "Priya",      "CALLS",              "mama"),
            ("Aarav", "Vijay",      "CALLS",              "papa"),
            ("Aarav", "Rohan",      "PLAYS_WITH",         "classmate"),
            ("Aarav", "Dr. Meera",  "SEES_FOR_THERAPY",   "SLP"),
            ("Aarav", "Dr. Sharma", "TRUSTS_FOR_MEDICAL", "physio"),
        ]
        for src, tgt, rel, label in relationships:
            self.run(f"""
                MATCH (a {{name: $src}}), (b {{name: $tgt}})
                CREATE (a)-[:{rel} {{label: $label}}]->(b)
            """, src=src, tgt=tgt, label=label)

        # ── ROUTINES ──────────────────────────────────────────
        routines = [
            dict(name="Morning Routine",
                 time="07:30-09:00",     location="home",
                 activities="wake_up,breakfast,physiotherapy,school_bus",
                 people_present="Priya,Vijay",
                 common_utterances="I am ready,more juice,my leg hurts",
                 context_key="morning"),
            dict(name="School Time",
                 time="09:00-14:00",
                 location="St. Mary's Special School",
                 activities="class,lunch,therapy,play",
                 people_present="Rohan,teachers",
                 common_utterances="I want to play,help me,I am tired",
                 context_key="school"),
            dict(name="Evening Routine",
                 time="16:00-20:00",     location="home",
                 activities="homework,play,dinner,bedtime",
                 people_present="Priya,Vijay",
                 common_utterances="I am hungry,I want to watch TV,good night mama",
                 context_key="evening"),
        ]
        for r in routines:
            self.run("""
                CREATE (:Routine {
                    name:               $name,
                    time:               $time,
                    location:           $location,
                    activities:         $activities,
                    people_present:     $people_present,
                    common_utterances:  $common_utterances,
                    context_key:        $context_key
                })
            """, **r)
            self.run("""
                MATCH (p:Patient {name: 'Aarav'}), (r:Routine {name: $name})
                CREATE (p)-[:HAS_ROUTINE]->(r)
            """, name=r["name"])

        # ── VOCABULARY ────────────────────────────────────────
        vocab = [
            ("hungry",     "hungry",          "want food,tummy hurts",      4.5, "high",   "meal_time,discomfort"),
            ("juice",      "juice",           "drink,orange juice",          3.0, "high",   "meal_time,morning"),
            ("tired",      "sleepy",          "tired,need rest",             4.0, "medium", "evening,school"),
            ("pain",       "my leg hurts",    "hurts,pain",                  5.0, "medium", "physio,morning"),
            ("play",       "I want to play",  "play time,let us play",       3.5, "high",   "school,evening"),
            ("ready",      "I am ready",      "all done,finished",           4.0, "high",   "morning,transition"),
            ("help",       "help me",         "I need help,please help",     3.8, "medium", "school,therapy"),
            ("happy",      "I am happy",      "feeling good,yay",            4.2, "low",    "general"),
            ("television", "I want TV",       "show,cartoon",                5.5, "medium", "evening"),
            ("night",      "good night",      "bye,sleep time",              3.5, "medium", "bedtime"),
        ]
        for word, preferred, alternatives, acq_age, freq, tags in vocab:
            self.run("""
                CREATE (:Vocabulary {
                    word:            $word,
                    preferred_form:  $preferred,
                    alternatives:    $alternatives,
                    acquisition_age: $acq_age,
                    frequency:       $freq,
                    context_tags:    $tags
                })
            """, word=word, preferred=preferred, alternatives=alternatives,
                acq_age=acq_age, freq=freq, tags=tags)
            self.run("""
                MATCH (p:Patient {name: 'Aarav'}), (v:Vocabulary {word: $word})
                CREATE (p)-[:PREFERS_VOCAB {frequency: $freq}]->(v)
            """, word=word, freq=freq)

        # ── EPISODES ──────────────────────────────────────────
        episodes = [
            ("ep_001", "2026-05-28", "morning",  "Priya",
             "more juice please,I am ready mama",           0.61, True),
            ("ep_002", "2026-05-27", "therapy",  "Dr. Meera",
             "I want to play,help me,I am tired",           0.59, True),
            ("ep_003", "2026-05-26", "school",   "Rohan",
             "let us play dinosaurs,I am happy",            0.63, True),
            ("ep_004", "2026-05-25", "physio",   "Dr. Sharma",
             "my leg hurts,I am tired",                     0.55, False),
            ("ep_005", "2026-05-24", "evening",  "Vijay",
             "I am hungry,I want TV,good night papa",       0.60, True),
        ]
        for ep_id, date, context, partner, utterances, mcds, success in episodes:
            self.run("""
                CREATE (:Episode {
                    ep_id:      $ep_id,
                    date:       $date,
                    context:    $context,
                    partner:    $partner,
                    utterances: $utterances,
                    mcds_score: $mcds,
                    successful: $success
                })
            """, ep_id=ep_id, date=date, context=context, partner=partner,
                utterances=utterances, mcds=mcds, success=success)
            self.run("""
                MATCH (p:Patient {name: 'Aarav'}), (e:Episode {ep_id: $ep_id})
                CREATE (p)-[:HAD_EPISODE]->(e)
            """, ep_id=ep_id)
            # Link episode to the partner Person node
            self.run("""
                MATCH (e:Episode {ep_id: $ep_id}), (person {name: $partner})
                CREATE (e)-[:INVOLVED]->(person)
            """, ep_id=ep_id, partner=partner)

        # ── SUMMARY ───────────────────────────────────────────
        counts = self.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count")
        total_nodes = sum(r["count"] for r in counts)
        total_edges = self.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        print(f"[Neo4j] Graph built: {total_nodes} nodes, {total_edges} relationships")
        for row in counts:
            print(f"         {row['label']:15s}: {row['count']}")

    # ── CONTEXT QUERIES ───────────────────────────────────────

    def query_context(self, current_context: str, partner: str) -> dict:
        """
        Runs four Cypher queries and returns a structured
        context dict ready for format_context_for_prompt().
        """
        context_key_map = {
            "morning": "morning",
            "physio":  "morning",   # physio happens in morning slot
            "school":  "school",
            "therapy": "school",    # therapy is during school time
            "evening": "evening",
        }
        resolved_key = context_key_map.get(current_context.lower(), current_context.lower())

        # 1. Patient profile
        patient_rows = self.run("""
            MATCH (p:Patient {name: 'Aarav'})
            RETURN p
        """)
        patient_data = patient_rows[0]["p"] if patient_rows else {}

        # 2. Routine
        routine_rows = self.run("""
            MATCH (:Patient {name: 'Aarav'})-[:HAS_ROUTINE]->(r:Routine)
            WHERE r.context_key = $key
            RETURN r
        """, key=resolved_key)
        routine_data = routine_rows[0]["r"] if routine_rows else {}

        # 3. Partner + relationship
        partner_rows = self.run("""
            MATCH (p:Patient {name: 'Aarav'})-[rel]->(person {name: $partner})
            RETURN person, type(rel) AS rel_type, rel.label AS rel_label
        """, partner=partner)
        partner_data  = partner_rows[0]["person"]   if partner_rows else {}
        rel_label     = partner_rows[0]["rel_label"] if partner_rows else ""

        # 4. Vocabulary (context-relevant + high frequency)
        vocab_rows = self.run("""
            MATCH (:Patient {name: 'Aarav'})-[:PREFERS_VOCAB]->(v:Vocabulary)
            WHERE v.context_tags CONTAINS $ctx OR v.frequency = 'high'
            RETURN v.word           AS word,
                   v.preferred_form AS preferred_form,
                   v.frequency      AS frequency
            ORDER BY
                CASE v.frequency
                    WHEN 'high'   THEN 1
                    WHEN 'medium' THEN 2
                    ELSE 3
                END
            LIMIT 8
        """, ctx=current_context.lower())

        # 5. Past successful episodes
        episode_rows = self.run("""
            MATCH (:Patient {name: 'Aarav'})-[:HAD_EPISODE]->(e:Episode)
            WHERE e.context = $ctx AND e.successful = true
            RETURN e.utterances AS utterances,
                   e.mcds_score AS mcds_score,
                   e.partner    AS partner
            ORDER BY e.mcds_score DESC
            LIMIT 3
        """, ctx=current_context.lower())

        return {
            "patient":       patient_data,
            "routine":       routine_data,
            "partner":       partner_data,
            "rel_label":     rel_label,
            "vocabulary":    vocab_rows,
            "past_episodes": episode_rows,
        }

    def add_episode(self, context: str, partner: str, utterances: str, mcds: float, success: bool, emotion: str):
        """Adds a new interaction episode to the graph."""
        import datetime
        ep_id = f"ep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        date  = datetime.datetime.now().strftime("%Y-%m-%d")
        
        self.run("""
            CREATE (e:Episode {
                ep_id:      $ep_id,
                date:       $date,
                context:    $context,
                partner:    $partner,
                utterances: $utterances,
                mcds_score: $mcds,
                successful: $success,
                emotion:    $emotion
            })
        """, ep_id=ep_id, date=date, context=context, partner=partner,
            utterances=utterances, mcds=mcds, success=success, emotion=emotion)
            
        self.run("""
            MATCH (p:Patient {name: 'Aarav'}), (e:Episode {ep_id: $ep_id})
            CREATE (p)-[:HAD_EPISODE]->(e)
        """, ep_id=ep_id)
        
        self.run("""
            MATCH (e:Episode {ep_id: $ep_id}), (person {name: $partner})
            CREATE (e)-[:INVOLVED]->(person)
        """, ep_id=ep_id, partner=partner)
        print(f"[Neo4j] Episode {ep_id} added (Emotion: {emotion}).")

    def get_summary(self) -> dict:
        """Returns node and relationship counts for display."""
        node_counts = self.run("""
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
        """)
        rel_counts = self.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            ORDER BY count DESC
        """)
        return {"nodes": node_counts, "relationships": rel_counts}


# ── CONTEXT FORMATTER ─────────────────────────────────────────

def format_context_for_prompt(context: dict) -> str:
    """
    Serialises the Neo4j context dict into a structured string
    injected as the LLM system prompt.
    """
    patient  = context.get("patient",       {})
    routine  = context.get("routine",        {})
    partner  = context.get("partner",        {})
    rel_lbl  = context.get("rel_label",      "")
    vocab    = context.get("vocabulary",     [])
    episodes = context.get("past_episodes",  [])

    lines = [
        "=== PATIENT KNOWLEDGE GRAPH CONTEXT ===",
        f"Patient name    : Aarav",
        f"Age             : {patient.get('age')} (chronological), "
        f"{patient.get('linguistic_age')} (linguistic)",
        f"CP subtype      : {patient.get('cp_subtype')}",
        f"MCDS baseline   : {patient.get('mcds_baseline')}",
        f"AAC device      : {patient.get('aac_device')}",
    ]

    if routine:
        lines += [
            "",
            f"Current routine : {routine.get('time', '')}",
            f"Location        : {routine.get('location', '')}",
            f"Activities      : {routine.get('activities', '')}",
            f"People present  : {routine.get('people_present', '')}",
            f"Common phrases  : {routine.get('common_utterances', '')}",
        ]

    if partner:
        lines += [
            "",
            f"Talking to      : {partner.get('role', '')} "
            f"(Aarav calls them '{rel_lbl or partner.get('nickname', '')}') ",
            f"Comm. style     : {partner.get('communication_style', '')}",
        ]

    if vocab:
        lines.append("")
        lines.append("Preferred vocabulary for this context:")
        for v in vocab:
            lines.append(
                f"  • '{v['word']}' → Aarav says: "
                f"\"{v['preferred_form']}\"  [{v['frequency']} frequency]"
            )

    if episodes:
        lines.append("")
        lines.append("Past successful utterances in similar contexts:")
        for ep in episodes:
            lines.append(f"  • \"{ep['utterances']}\"  (MCDS: {ep['mcds_score']})")

    lines.append("=========================================")
    return "\n".join(lines)


# ── BUILD ON DIRECT EXECUTION ─────────────────────────────────

if __name__ == "__main__":
    kg = Neo4jKG()
    kg.build()
    kg.close()
    print("\n[Done] Open http://localhost:7474 and run:")
    print("       MATCH (n) RETURN n")
    print("       to visualise the full graph.")
