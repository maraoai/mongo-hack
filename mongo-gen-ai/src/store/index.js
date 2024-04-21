import { createStore } from 'vuex'
import global from './modules/global'
export default createStore({
  modules: {
    global,
  }
})