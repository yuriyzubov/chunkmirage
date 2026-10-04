import"./modulepreload-polyfill-P2Xu9kJm.js";import{t as e}from"./cards-CdoSt5Io.js";var t=document.getElementById(`cards`);for(let n of e){let e=n.kind===`pipeline`?`pipeline.html?card=${n.id}`:n.kind===`map`?`map.html?card=${n.id}`:n.kind===`link`?n.href:null,r=document.createElement(`article`);if(r.className=`card`,r.innerHTML=`
    <a href=""><img alt="" loading="lazy"></a>
    <div class="body">
      <h2></h2>
      <p class="blurb"></p>
      <p class="data"></p>
      <details><summary>The same from Python</summary><pre></pre></details>
      <div class="actions"><a class="open" href="">Open</a></div>
    </div>`,e)for(let t of r.querySelectorAll(`a`))t.href=e;else r.querySelector(`img`).parentElement.replaceWith(r.querySelector(`img`)),r.querySelector(`details`).open=!0,r.querySelector(`summary`).textContent=`Run it from Python`,r.querySelector(`.actions`).innerHTML=`<p class="why"></p>`,r.querySelector(`.why`).textContent=n.why??``;r.querySelector(`img`).src=n.image,r.querySelector(`h2`).textContent=n.title,r.querySelector(`.blurb`).textContent=n.blurb,r.querySelector(`.data`).textContent=`Data: ${n.data}`,r.querySelector(`pre`).textContent=n.command,t.append(r)}