import { useEffect } from 'react'

import '../tailwind.css'

import type { AppProps } from 'next/app'

export default function App({ Component, pageProps }: AppProps) {
  useEffect(() => {
    const init = async () => {
      const { default: eruda } = await import('eruda')
      eruda.init()
    }
    init()
  }, [])

  return <Component {...pageProps} />
}
