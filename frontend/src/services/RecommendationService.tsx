import { apiRequest } from "./ApiService";
import type { Book } from "@/hooks/useBooks";

export async function getRecommendations(limit = 10): Promise<Book[]> {
	return apiRequest<Book[]>(`/recommendations?limit=${limit}`);
}
