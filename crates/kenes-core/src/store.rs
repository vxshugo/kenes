//! SQLite persistence for meetings, final transcript segments, notes and settings.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use kenes_types::{Segment, Source};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS meetings (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    context    TEXT NOT NULL DEFAULT '',
    started_at TEXT NOT NULL,
    ended_at   TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    id         TEXT NOT NULL,
    source     TEXT NOT NULL,
    speaker    TEXT,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (meeting_id, id)
);
CREATE INDEX IF NOT EXISTS segments_by_time ON segments(meeting_id, start_ms);
CREATE TABLE IF NOT EXISTS notes (
    id         TEXT PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    content    TEXT NOT NULL,
    trigger    TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS speaker_names (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    label      TEXT NOT NULL,
    name       TEXT NOT NULL,
    PRIMARY KEY (meeting_id, label)
);
CREATE TABLE IF NOT EXISTS segment_embeddings (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    segment_id TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    PRIMARY KEY (meeting_id, segment_id)
);
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"#;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingSummary {
    pub id: String,
    pub title: String,
    pub started_at: String,
    pub ended_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Note {
    pub id: String,
    pub meeting_id: String,
    pub kind: String,
    pub content: String,
    pub trigger: Option<String>,
    pub created_at: String,
}

/// A speaker label seen in a meeting, with the name the user gave it (if any).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Speaker {
    pub label: String,
    pub name: Option<String>,
    pub segment_count: u64,
    pub talk_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Meeting {
    #[serde(flatten)]
    pub summary: MeetingSummary,
    pub context: String,
    pub segments: Vec<Segment>,
    pub notes: Vec<Note>,
    pub speakers: Vec<Speaker>,
}

pub struct Store {
    conn: Mutex<Connection>,
}

pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;")?;
        conn.execute_batch(SCHEMA).context("applying schema")?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        // A panic while holding the lock can't leave SQLite half-written, so keep going.
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn create_meeting(&self, title: &str, context: &str) -> Result<MeetingSummary> {
        let m = MeetingSummary {
            id: uuid::Uuid::new_v4().to_string(),
            title: title.to_owned(),
            started_at: now_iso(),
            ended_at: None,
        };
        self.conn().execute(
            "INSERT INTO meetings (id, title, context, started_at) VALUES (?1, ?2, ?3, ?4)",
            params![m.id, m.title, context, m.started_at],
        )?;
        Ok(m)
    }

    pub fn end_meeting(&self, id: &str) -> Result<()> {
        self.conn().execute(
            "UPDATE meetings SET ended_at = ?2 WHERE id = ?1 AND ended_at IS NULL",
            params![id, now_iso()],
        )?;
        Ok(())
    }

    pub fn list_meetings(&self) -> Result<Vec<MeetingSummary>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT id, title, started_at, ended_at FROM meetings ORDER BY started_at DESC, rowid DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(MeetingSummary {
                id: r.get(0)?,
                title: r.get(1)?,
                started_at: r.get(2)?,
                ended_at: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn get_meeting(&self, id: &str) -> Result<Option<Meeting>> {
        let Some((summary, context, segments, notes)) = self.meeting_parts(id)? else {
            return Ok(None);
        };
        let speakers = self.list_speakers(id)?;
        Ok(Some(Meeting {
            summary,
            context,
            segments,
            notes,
            speakers,
        }))
    }

    #[allow(clippy::type_complexity)]
    fn meeting_parts(
        &self,
        id: &str,
    ) -> Result<Option<(MeetingSummary, String, Vec<Segment>, Vec<Note>)>> {
        let conn = self.conn();
        let head = conn
            .query_row(
                "SELECT id, title, started_at, ended_at, context FROM meetings WHERE id = ?1",
                [id],
                |r| {
                    Ok((
                        MeetingSummary {
                            id: r.get(0)?,
                            title: r.get(1)?,
                            started_at: r.get(2)?,
                            ended_at: r.get(3)?,
                        },
                        r.get::<_, String>(4)?,
                    ))
                },
            )
            .optional()?;
        let Some((summary, context)) = head else {
            return Ok(None);
        };

        let mut stmt = conn.prepare(
            "SELECT id, source, speaker, start_ms, end_ms, text FROM segments
             WHERE meeting_id = ?1 ORDER BY start_ms, id",
        )?;
        let segments = stmt
            .query_map([id], |r| {
                let source: String = r.get(1)?;
                Ok(Segment {
                    id: r.get(0)?,
                    source: if source == "mic" {
                        Source::Mic
                    } else {
                        Source::System
                    },
                    speaker: r.get(2)?,
                    start_ms: r.get::<_, i64>(3)? as u64,
                    end_ms: r.get::<_, i64>(4)? as u64,
                    text: r.get(5)?,
                    is_final: true,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut stmt = conn.prepare(
            "SELECT id, meeting_id, kind, content, trigger, created_at FROM notes
             WHERE meeting_id = ?1 ORDER BY created_at, rowid",
        )?;
        let notes = stmt
            .query_map([id], |r| {
                Ok(Note {
                    id: r.get(0)?,
                    meeting_id: r.get(1)?,
                    kind: r.get(2)?,
                    content: r.get(3)?,
                    trigger: r.get(4)?,
                    created_at: r.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(Some((summary, context, segments, notes)))
    }

    pub fn delete_meeting(&self, id: &str) -> Result<()> {
        self.conn()
            .execute("DELETE FROM meetings WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Stores a final segment. Partials and empty finals (retracted noise) are never persisted.
    pub fn save_segment(&self, meeting_id: &str, seg: &Segment) -> Result<()> {
        if seg.text.trim().is_empty() {
            return Ok(());
        }
        self.conn().execute(
            "INSERT OR REPLACE INTO segments (meeting_id, id, source, speaker, start_ms, end_ms, text)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                meeting_id,
                seg.id,
                seg.source.as_str(),
                seg.speaker,
                seg.start_ms as i64,
                seg.end_ms as i64,
                seg.text
            ],
        )?;
        Ok(())
    }

    /// Every label used in the meeting (plus named labels with no segments left), by talk time.
    pub fn list_speakers(&self, meeting_id: &str) -> Result<Vec<Speaker>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT label, MAX(name), SUM(cnt), SUM(talk) FROM (
                 SELECT speaker AS label, NULL AS name, COUNT(*) AS cnt, SUM(end_ms - start_ms) AS talk
                 FROM segments WHERE meeting_id = ?1 AND speaker IS NOT NULL GROUP BY speaker
                 UNION ALL
                 SELECT label, name, 0, 0 FROM speaker_names WHERE meeting_id = ?1
             ) GROUP BY label ORDER BY SUM(talk) DESC, label",
        )?;
        let rows = stmt.query_map([meeting_id], |r| {
            Ok(Speaker {
                label: r.get(0)?,
                name: r.get(1)?,
                segment_count: r.get::<_, i64>(2)? as u64,
                talk_ms: r.get::<_, i64>(3)? as u64,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Names a speaker label for one meeting; an empty name removes it.
    pub fn rename_speaker(&self, meeting_id: &str, label: &str, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            self.conn().execute(
                "DELETE FROM speaker_names WHERE meeting_id = ?1 AND label = ?2",
                params![meeting_id, label],
            )?;
        } else {
            self.conn().execute(
                "INSERT INTO speaker_names (meeting_id, label, name) VALUES (?1, ?2, ?3)
                 ON CONFLICT(meeting_id, label) DO UPDATE SET name = excluded.name",
                params![meeting_id, label, name],
            )?;
        }
        Ok(())
    }

    pub fn save_embedding(
        &self,
        meeting_id: &str,
        segment_id: &str,
        embedding: &[f32],
    ) -> Result<()> {
        let blob: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.conn().execute(
            "INSERT OR REPLACE INTO segment_embeddings (meeting_id, segment_id, embedding) VALUES (?1, ?2, ?3)",
            params![meeting_id, segment_id, blob],
        )?;
        Ok(())
    }

    /// Stored embeddings keyed by segment id.
    pub fn load_embeddings(
        &self,
        meeting_id: &str,
    ) -> Result<std::collections::HashMap<String, Vec<f32>>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT segment_id, embedding FROM segment_embeddings WHERE meeting_id = ?1",
        )?;
        let rows = stmt.query_map([meeting_id], |r| {
            let blob: Vec<u8> = r.get(1)?;
            let emb = blob
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            Ok((r.get::<_, String>(0)?, emb))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn set_segment_speakers(
        &self,
        meeting_id: &str,
        changes: &[(String, Option<String>)],
    ) -> Result<()> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        {
            let mut stmt =
                tx.prepare("UPDATE segments SET speaker = ?3 WHERE meeting_id = ?1 AND id = ?2")?;
            for (segment_id, speaker) in changes {
                stmt.execute(params![meeting_id, segment_id, speaker])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn save_note(
        &self,
        meeting_id: &str,
        kind: &str,
        content: &str,
        trigger: Option<&str>,
    ) -> Result<String> {
        let id = uuid::Uuid::new_v4().to_string();
        self.conn().execute(
            "INSERT INTO notes (id, meeting_id, kind, content, trigger, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![id, meeting_id, kind, content, trigger, now_iso()],
        )?;
        Ok(id)
    }

    pub fn get_kv(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn()
            .query_row("SELECT value FROM kv WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_kv(&self, key: &str, value: &str) -> Result<()> {
        self.conn().execute(
            "INSERT INTO kv (key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: &str, source: Source, start: u64, text: &str) -> Segment {
        Segment {
            id: id.into(),
            source,
            speaker: None,
            start_ms: start,
            end_ms: start + 1000,
            text: text.into(),
            is_final: true,
        }
    }

    #[test]
    fn meeting_round_trip() {
        let store = Store::open_in_memory().unwrap();
        let m = store.create_meeting("Синк", "повестка").unwrap();
        store
            .save_segment(
                &m.id,
                &seg("system-1", Source::System, 2000, "қашан бітеді"),
            )
            .unwrap();
        store
            .save_segment(&m.id, &seg("mic-1", Source::Mic, 500, "всем привет"))
            .unwrap();
        // Re-saving the same id replaces rather than duplicates.
        store
            .save_segment(
                &m.id,
                &seg("mic-1", Source::Mic, 500, "всем привет коллеги"),
            )
            .unwrap();
        let note_id = store
            .save_note(&m.id, "hint", "- ответ", Some("question"))
            .unwrap();
        store.end_meeting(&m.id).unwrap();

        let got = store.get_meeting(&m.id).unwrap().unwrap();
        assert_eq!(got.context, "повестка");
        assert!(got.summary.ended_at.is_some());
        let texts: Vec<_> = got.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, ["всем привет коллеги", "қашан бітеді"]);
        assert_eq!(got.segments[1].source, Source::System);
        assert_eq!(got.notes.len(), 1);
        assert_eq!(got.notes[0].id, note_id);
        assert_eq!(got.notes[0].trigger.as_deref(), Some("question"));
    }

    #[test]
    fn speakers_names_and_embeddings() {
        let store = Store::open_in_memory().unwrap();
        let m = store.create_meeting("x", "").unwrap();
        let mut a = seg("system-1", Source::System, 0, "a");
        a.speaker = Some("sys:1".into());
        let mut b = seg("system-2", Source::System, 2000, "b");
        b.speaker = Some("sys:2".into());
        let mut c = seg("system-3", Source::System, 4000, "c");
        c.speaker = Some("sys:1".into());
        for s in [&a, &b, &c] {
            store.save_segment(&m.id, s).unwrap();
        }
        store.rename_speaker(&m.id, "sys:2", "Айдос").unwrap();
        store.rename_speaker(&m.id, "sys:9", "Никто").unwrap();

        let sp = store.list_speakers(&m.id).unwrap();
        assert_eq!(
            sp[0],
            Speaker {
                label: "sys:1".into(),
                name: None,
                segment_count: 2,
                talk_ms: 2000
            }
        );
        assert_eq!(
            sp[1],
            Speaker {
                label: "sys:2".into(),
                name: Some("Айдос".into()),
                segment_count: 1,
                talk_ms: 1000
            }
        );
        assert_eq!(sp[2].label, "sys:9");
        store.rename_speaker(&m.id, "sys:9", " ").unwrap();
        assert_eq!(store.list_speakers(&m.id).unwrap().len(), 2);

        store
            .save_embedding(&m.id, "system-1", &[0.5, -1.25, 3.0])
            .unwrap();
        assert_eq!(
            store.load_embeddings(&m.id).unwrap()["system-1"],
            vec![0.5, -1.25, 3.0]
        );

        store
            .set_segment_speakers(&m.id, &[("system-3".into(), Some("sys:2".into()))])
            .unwrap();
        let got = store.get_meeting(&m.id).unwrap().unwrap();
        assert_eq!(got.segments[2].speaker.as_deref(), Some("sys:2"));
        assert_eq!(got.speakers[0].label, "sys:2");
    }

    #[test]
    fn delete_cascades() {
        let store = Store::open_in_memory().unwrap();
        let m = store.create_meeting("x", "").unwrap();
        store
            .save_segment(&m.id, &seg("mic-1", Source::Mic, 0, "a"))
            .unwrap();
        store.save_note(&m.id, "summary", "b", None).unwrap();
        store.delete_meeting(&m.id).unwrap();
        assert!(store.get_meeting(&m.id).unwrap().is_none());
        let n: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM segments", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn delete_cascades_to_every_child_table() {
        let store = Store::open_in_memory().unwrap();
        let m = store.create_meeting("x", "").unwrap();
        let keep = store.create_meeting("y", "").unwrap();
        for id in [&m.id, &keep.id] {
            store
                .save_segment(id, &seg("mic-1", Source::Mic, 0, "a"))
                .unwrap();
            store.save_note(id, "summary", "b", None).unwrap();
            store.rename_speaker(id, "sys:1", "Айдос").unwrap();
            store.save_embedding(id, "mic-1", &[1.0, 0.0]).unwrap();
        }
        store.delete_meeting(&m.id).unwrap();
        for table in ["segments", "notes", "speaker_names", "segment_embeddings"] {
            let (gone, kept): (i64, i64) = store
                .conn()
                .query_row(
                    &format!("SELECT SUM(meeting_id = ?1), SUM(meeting_id = ?2) FROM {table}"),
                    params![m.id, keep.id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!((gone, kept), (0, 1), "{table}");
        }
    }

    #[test]
    fn kv_upsert() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.get_kv("k").unwrap(), None);
        store.set_kv("k", "1").unwrap();
        store.set_kv("k", "2").unwrap();
        assert_eq!(store.get_kv("k").unwrap().as_deref(), Some("2"));
    }

    #[test]
    fn meetings_listed_newest_first() {
        let store = Store::open_in_memory().unwrap();
        let a = store.create_meeting("a", "").unwrap();
        let b = store.create_meeting("b", "").unwrap();
        let ids: Vec<_> = store
            .list_meetings()
            .unwrap()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(ids, [b.id, a.id]);
    }
}
