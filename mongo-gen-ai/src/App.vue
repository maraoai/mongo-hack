<template>
  <div :key="key">
    <h1 style="text-align: center; color:pink">Thinky Space</h1>
    <div class="flex flex-row">
      <div class="min-h-screen w-2/4 overflow-visible">
        <div class="grid grid-cols-1 gap-4">
          <div class="markdown">
            <Markdown :source="textToDisplay" />
          </div>
        </div>
      </div>
      <div class="w-2/4">
        <div v-if="loaded" class="chart-container" style="position: relative; height:60vh; width:100vw; text-align: center;">
          <Scatter :key="key" id="chart"  :data="data" :options="options" @click="handleChartClick"/>
        </div>
        <div v-else><p>Hang tight!</p></div>
        <div>
          <form>   
          <label for="search" class="mb-2 text-sm font-medium text-gray-900 sr-only dark:text-white">Search</label>
          <div class="relative flex flex-row items-center align-middle justify-center w-full">
              <div class="absolute inset-y-0 start-0 flex items-center ps-3 pointer-events-none">
                  <svg class="w-4 h-4 text-gray-500 dark:text-gray-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 20">
                      <path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"/>
                  </svg>
              </div>
              <input v-model="searchText" type="search" id="search" class="block w-full p-4 ps-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="Search" required />
              <button @click.prevent="similaritySearch(searchText)" type="submit" class="text-white absolute end-2.5 bottom-2.5 bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800">Search</button>

            </div>
        </form>
        </div>
      </div>
    </div>
    
  </div>
</template>

<script >
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
} from 'chart.js';
import { Scatter } from 'vue-chartjs';
import { mapGetters } from "vuex";
import store from './store';
import axios from 'axios'
import { nextTick } from 'vue'
import Markdown from 'vue3-markdown-it';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);
export default {
  components: {
    Scatter,
    Markdown
  },
  mounted() {
  },
  data() {
    return {
      key: 1,
      viewedDocument: '',
      searchText: '',
      loaded: true,
      options: {
        // onClick: function(evt) {   
        //   var element = this.getElementAtEvent(evt);
        //   debugger
        //   if(element.length > 0)
        //   {
        //     var ind = element[0]._index;
        //   }
        // },
        hover: {
          onHover: function(e) {
            var point = this.getElementAtEvent(e);
            if (point.length) e.target.style.cursor = 'pointer';
            else e.target.style.cursor = 'default';
          }
        },
        plugins: {
          tooltip: {
            callbacks: {
              label: async function(ctx) {
                console.log(ctx);
                let label = ctx.label
                label += " (" + ctx.raw.text.slice(0, 9)+ ")";
                const idx = ctx.dataIndex
                const doc = ctx.dataset.data[idx].text
                this.viewedDocument = doc
                this.key = this.key == 1 ? 0 : 1
                console.log(this.viewedDocument, ' hello')
                store.commit("global/setTextToDisplay", doc)
                await nextTick()
                return label;
              },
              onClick: function (ctx) {
                console.log('hell o world')
              }
            }
          }
        },
        scales: {
          x: {
            suggestedMin: -1,
            suggestedMax: 1,
            ticks: {
              display: false
            },
            grid: {
              display: false
            }
          },
          y: {
            suggestedMin: -1,
            suggestedMax: 1,
            ticks: {
              display: false
            },
            grid: {
              display: false
            }
          }
      }
      },
      data: {
        datasets: [
          {
            label: 'Search Term',
            fill: false,
            borderColor: '#7acbf9',
            backgroundColor: '#7acbf9',
            data: [
            ],
          },
          {
            label: 'Close Documents',
            fill: false,
            borderColor: '#f87979',
            backgroundColor: '#f87979',
            data: [
              
            ],
          },
        ],
      }
    }
  },
  methods: {
    handleChartClick(event, chartElements) {
      this.viewedDocument = 'world'
      console.log('hi ', this.viewedDocument)
      // if (chartElements.length > 0) {
      //   const clickedPoint = chartElements[0];
      //   const datasetIndex = clickedPoint._datasetIndex;
      //   const index = clickedPoint._index;
      //   const point = this.scatterData.datasets[datasetIndex].data[index];
      //   this.handlePointClick(point);
      // }
    },
    handlePointClick(point) {
      console.log('Point clicked:', point);
      // Call your method here with the clicked point data
    },
    async similaritySearch (text) {
      // this.loaded = false
      const url = 'https://cpfiffer--mongo-hack-handle-request.modal.run'
      const res = await fetch(url,
      {
        method: "POST",
        body: JSON.stringify({text: text}),
        headers: {
          "Content-Type": "application/json",
        },
      })
      const resStrObj = await res.json();
      const resObj = JSON.parse(resStrObj)

      this.data.datasets[1].data = resObj.data
      this.data.datasets[0].data = [
      resObj.data[0]
    ]
      this.key = this.key == 1 ? 0 : 1
      this.loaded = true
    },
  },
  computed: {
    ...mapGetters({
      textToDisplay: "global/loadTextToDisplay"
    })
  }
}
</script>

<style scoped="scss">

</style>
