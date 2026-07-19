export default function Footer() {
    return (
        <footer className="mt-12 border-t border-gray-700 bg-black px-4 py-8 text-white sm:px-6 lg:px-8">
            <div className="mx-auto max-w-7xl">
                <div className="mb-8 grid grid-cols-1 gap-8 md:grid-cols-3">
                    <div>
                        <h3 className="font-bold text-lg mb-3">About E-Library</h3>
                        <p className="text-gray-400 text-sm">A platform for discovering and exploring books.</p>
                    </div>
                </div>
                <div className="border-t border-gray-700 pt-6 text-center">
                    <p className="text-gray-400 text-sm">&copy; 2026 E-Library</p>
                </div>
            </div>
        </footer>
    );
}