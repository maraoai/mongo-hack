
const state = {
  text: ''
}

const mutations = {
  setTextToDisplay (state, text) {
    state.text = text
  }
}

const actions = {
  setText (context, text) {
    context.commit('setTextToDisplay', text)
  }
}

const getters = {
  loadTextToDisplay: (state) => {
    return state.text
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions,
  getters
}
