"use strict";
const $ = id => document.getElementById(id);
let current = null, offset = 0, map = null, markers = [], mapLibraries = null;
const pageSize = 60;
const dateTime = value => new Date(value).toLocaleString("fr-FR", {dateStyle:"short", timeStyle:"short"});
function localInput(iso) { const d = new Date(iso); return new Date(d.getTime() - d.getTimezoneOffset()*60000).toISOString().slice(0,16); }
function element(tag, content, className) { const el = document.createElement(tag); if(content !== undefined) el.textContent = content; if(className) el.className = className; return el; }
function link(label, href) { const a = element("a", label); const u = new URL(href); if (!["http:","https:"].includes(u.protocol)) return element("span",label); a.href=u.href; a.target="_blank"; a.rel="noopener noreferrer"; return a; }
function render() {
  $("items").replaceChildren();
  for (const item of current.items) {
    const card = element("article", undefined, "card");
    for(const c of item.categories) card.append(element("span", current.categories[c], "tag"));
    card.append(element("h2",item.title));
    const price = {free:"Gratuit", paid:"Payant — voir le tarif", conditional_free:"Gratuit sous conditions",unknown:"Tarif à vérifier"}[item.price_type];
    card.append(element("p",price,"price"),element("p",item.lead_text || item.description.slice(0,230)));
    const session=item.next_occurrence;
    card.append(element("p",dateTime(session.start_utc)+" → "+dateTime(session.end_utc)));
    card.append(element("p",item.locations.map(l=>l.venue_name || l.street).join(" · ")));
    if(item.quality_flags.includes("multi_location_schedule_unassigned")) card.append(element("p","Plusieurs lieux : vérifier l’adresse de la séance.","hint"));
    if(item.source_url) card.append(link("Voir l’annonce",item.source_url));
    const point=item.locations[0];
    card.append(link("Google Maps","https://www.google.com/maps/search/?api=1&query="+encodeURIComponent(point.latitude+","+point.longitude)));
    $("items").append(card);
  }
  $("status").textContent = current.total ? `${current.total} activités · ${offset+1}–${offset+current.items.length} affichées` : "Aucune activité. Essaie un créneau plus large ou retire un filtre.";
  $("previous").disabled=offset===0; $("next").disabled=offset+current.items.length>=current.total;
  if(map) renderMarkers();
}
async function search(initial=false) {
  $("status").textContent="Recherche…";
  try {
    const params=new URLSearchParams({limit:pageSize,offset});
    if(!initial) {
      params.set("start_utc",new Date($("start").value).toISOString()); params.set("end_utc",new Date($("end").value).toISOString());
      if($("district").value) params.set("arrondissement",$("district").value);
      if($("budget").value!=="") params.set("budget_eur",$("budget").value);
      if($("category").value) params.set("category",$("category").value);
    }
    const response=await fetch("/api/catalog?"+params); const data=await response.json(); if(!response.ok) throw Error(data.error);
    current=data;
    if(initial) {
      $("start").value=localInput(data.snapshot_window.as_of_utc); $("end").value=localInput(data.snapshot_window.window_end_utc);
      for(const [id,label] of Object.entries(data.categories)) { const option=element("option",label); option.value=id; $("category").append(option); }
      $("snapshot").textContent="Catalogue figé du "+dateTime(data.snapshot_window.as_of_utc)+" au "+dateTime(data.snapshot_window.window_end_utc)+". Les horaires affichés utilisent le fuseau de ton navigateur.";
    }
    render();
  } catch(error) { $("status").textContent="Impossible de charger : "+error.message; }
}
function renderMarkers() {
  for(const marker of markers) marker.map=null; markers=[];
  const bounds=new google.maps.LatLngBounds();
  for(const item of current.items) for(const p of item.locations) {
    const marker=new mapLibraries.AdvancedMarkerElement({map,position:{lat:p.latitude,lng:p.longitude},title:item.title});
    marker.addListener("click",()=>{ const content=element("div"); content.append(element("strong",item.title),element("p",p.venue_name || p.street)); if(item.source_url) content.append(link("Voir l’annonce",item.source_url)); new google.maps.InfoWindow({content}).open({map,anchor:marker}); });
    markers.push(marker); bounds.extend(marker.position);
  }
  if(markers.length) map.fitBounds(bounds);
}
window.gm_authFailure=()=>{ $("map-status").textContent="Google Maps n’a pas accepté la clé. La liste reste disponible."; $("map").hidden=true; };
window.initSwiParisMap=async()=>{
  try { const {Map}=await google.maps.importLibrary("maps"); mapLibraries=await google.maps.importLibrary("marker"); $("map").hidden=false; map=new Map($("map"),{center:{lat:48.8566,lng:2.3522},zoom:12,maxZoom:17,mapId:window.swiMapId}); renderMarkers(); $("maps").disabled=false; $("map-status").textContent="Carte des activités de la page affichée. Source des lieux : Paris Data."; }
  catch(error) { $("map-status").textContent="Carte indisponible : "+error.message; }
};
$("maps").addEventListener("click",async()=>{
  if(!current) return;
  if(map) { $("map").hidden=!$("map").hidden; return; }
  try {
    const config=await (await fetch("/api/map-config")).json();
    if(!config.api_key) { $("map-status").textContent="Clé Google non configurée. La liste et les liens Google Maps fonctionnent sans clé."; return; }
    window.swiMapId=config.map_id;
    $("maps").disabled=true; $("map-status").textContent="Chargement de Google Maps…";
    const script=document.createElement("script");
    script.src="https://maps.googleapis.com/maps/api/js?"+new URLSearchParams({key:config.api_key,v:"quarterly",loading:"async",callback:"initSwiParisMap"}); script.async=true;
    script.onerror=()=>{ $("maps").disabled=false; $("map-status").textContent="Google Maps inaccessible. La liste reste disponible."; };
    document.head.append(script);
  } catch(error) { $("map-status").textContent=error.message; }
});
for(let i=1;i<=20;i++) { const option=element("option",String(i)); option.value=i; $("district").append(option); }
$("filters").addEventListener("submit",event=>{event.preventDefault();offset=0;search();});
$("previous").addEventListener("click",()=>{offset=Math.max(0,offset-pageSize);search();});
$("next").addEventListener("click",()=>{offset+=pageSize;search();});
search(true);
