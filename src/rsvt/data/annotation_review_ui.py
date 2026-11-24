"""Notebook form for the verified review session; no automatic decisions."""

from html import escape

import ipywidgets as widgets
from IPython.display import HTML, clear_output, display


class AnnotationReviewForm:
    def __init__(self, session):
        self.session = session
        self.active = None
        self._updating = False
        self._baseline = None
        self.point = widgets.Dropdown(description="Titik:", layout=widgets.Layout(width="95%"))
        self.status = widgets.Dropdown(description="Keputusan:", options=[("Belum dipilih",None),
            ("accepted","accepted"),("ambiguous","ambiguous"),("rejected","rejected")])
        self.hypothesis = widgets.Dropdown(description="Hipotesis:", layout=widgets.Layout(width="95%"))
        self.reviewer = widgets.Text(description="Reviewer:", placeholder="Nama reviewer",layout=widgets.Layout(width="95%"))
        self.evidence = widgets.Textarea(description="Bukti:",placeholder="Alasan keputusan atau ketidakpastian",
                                         layout=widgets.Layout(width="95%",height="85px"))
        self.save_button = widgets.Button(description="Simpan & Berikutnya",button_style="success",
                                           layout=widgets.Layout(width="210px"))
        self.discard_button = widgets.Button(description="Batalkan perubahan",layout=widgets.Layout(width="190px"))
        self.progress = widgets.HTML()
        self.message = widgets.HTML()
        self.preview = widgets.Output()
        self.root = widgets.VBox([self.progress,self.point,self.preview,self.status,self.hypothesis,
            self.reviewer,self.evidence,widgets.HBox([self.save_button,self.discard_button]),self.message])
        self.point.observe(self._point_changed,names="value")
        self.status.observe(self._status_changed,names="value")
        self.save_button.on_click(self._save_clicked)
        self.discard_button.on_click(self._discard_clicked)
        self._refresh_options()
        self._show(session.active_point_id)

    def _fields(self):
        return (self.status.value,self.hypothesis.value,self.reviewer.value,self.evidence.value)

    @property
    def dirty(self):
        return self._baseline is not None and self._fields()!=self._baseline

    def _notify(self,text,error=False):
        self.message.value=f'<p style="color:{"#b42318" if error else "#067647"}">{escape(text)}</p>'

    def _refresh_options(self):
        counts={k:0 for k in ("accepted","ambiguous","rejected","pending")}
        decisions=self.session.decisions
        options=[]
        for i,p in enumerate(self.session.source.points):
            status=decisions.get(p["point_id"],{}).get("status","pending")
            counts[status]+=1
            options.append((f"{i+1}/{len(self.session.source.points)} | {p['case_key']} | {p['label']} | {status}",p["point_id"]))
        self._updating=True
        self.point.options=options
        if self.active: self.point.value=self.active
        self._updating=False
        self.progress.value=("<b>Progres review</b>: " + " · ".join(f"{k}: {v}" for k,v in counts.items())
            + "<br>Koordinat ambigu tetap belum tervalidasi. Identitas sumber belum terverifikasi; training_ready=false.")

    def _show(self,point_id):
        # Verify before rendering or navigating; corrupted HTML is never displayed.
        html=self.session.source.preview(point_id)
        point=self.session.source.by_id[point_id]
        decision=self.session.decisions.get(point_id,{})
        retained_reviewer=self.reviewer.value
        self._updating=True
        self.point.value=point_id
        self.hypothesis.options=[("Pilih hipotesis",None)]+[(h["hypothesis"],h["hypothesis"])
            for h in point["hypotheses"] if h["status"]=="review_pending"]
        self.status.value=decision.get("status")
        self.hypothesis.value=decision.get("selected_hypothesis")
        self.hypothesis.disabled=self.status.value!="accepted"
        self.reviewer.value=decision.get("reviewer",retained_reviewer)
        self.evidence.value=decision.get("evidence","")
        self.active=point_id
        self._baseline=self._fields()
        self._updating=False
        with self.preview:
            clear_output(wait=True)
            display(HTML(html))

    def _point_changed(self,change):
        if self._updating or self.active is None: return
        if self.dirty:
            self._updating=True;self.point.value=self.active;self._updating=False
            self._notify("Simpan keputusan atau batalkan perubahan sebelum berpindah titik.",True)
            return
        try:
            self._show(change["new"])
            self.message.value=""
        except Exception as exc:
            self._updating=True;self.point.value=self.active;self._updating=False
            self._notify("Preview gagal diverifikasi: "+str(exc),True)

    def _status_changed(self,change):
        if self._updating: return
        self.hypothesis.disabled=change["new"]!="accepted"
        if self.hypothesis.disabled: self.hypothesis.value=None

    def _discard_clicked(self,_):
        try:
            self._show(self.active)
            self._notify("Perubahan formulir dibatalkan; keputusan tersimpan tetap dipertahankan.")
        except Exception as exc:
            self._notify("Preview gagal diverifikasi: "+str(exc),True)

    def _save_clicked(self,_):
        self.message.value=""
        self.save_button.disabled=True
        decision={"point_id":self.active,"status":self.status.value,
                  "selected_hypothesis":self.hypothesis.value,"reviewer":self.reviewer.value.strip(),
                  "evidence":self.evidence.value.strip()}
        try:
            self.session.save(decision)
        except Exception as exc:
            self._notify("Gagal menyimpan; titik dan isian tetap aktif: "+str(exc),True)
        else:
            self._refresh_options()
            try:
                self._show(self.session.active_point_id)
            except Exception as exc:
                self._notify("Keputusan tersimpan dan terverifikasi; preview berikutnya gagal diverifikasi: "+str(exc),True)
            else:
                self._notify("Keputusan tersimpan dan dibaca ulang dari Google Drive.")
        finally:
            self.save_button.disabled=False
