import Header from "./Header"
import Navbar from "./Navbar"
import Footer from './Footer'

export default function MainLayout({ children }: { children: React.ReactNode }) {
    return (
        <div className="min-h-screen flex flex-col bg-white dark:bg-black">
            <Header />
            <Navbar />
            <main className="grow w-full max-w-7xl mx-auto px-4 py-6 sm:px-6 sm:py-8 lg:px-8 lg:py-12">
                {children}
            </main>
            <Footer />
        </div>
    )
}